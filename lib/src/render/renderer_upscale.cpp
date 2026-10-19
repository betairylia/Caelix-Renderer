// The upscaler glue of the renderer (docs/DLSS.md): the render size query, the upscaler's
// lifetime (created per UpscalerDesc, the old one retired until its last frame is safe) and the
// upscaler info. The DLSS implementation is in dlss_upscaler.cpp; the per-frame pass sequence in
// renderer_render.cpp.
#include "render/renderer.h"

#include "core/log.h"
#include "render/renderer_internal.h"

#if CAELIX_DLSS
#include "render/dlss_upscaler.h"
#endif

#include <memory>
#include <utility>

namespace caelix {

CaelixResult Renderer::QueryUpscaler(uint32_t upscaler, uint32_t quality, uint32_t output_width,
                                     uint32_t output_height, uint32_t& render_width, uint32_t& render_height)
{
#if CAELIX_DLSS
    UpscaleQuery& query = upscale_query_;
    if (!query.valid || query.upscaler != upscaler || query.quality != quality ||
        query.output_width != output_width || query.output_height != output_height)
    {
        query = UpscaleQuery{};
        query.upscaler = upscaler;
        query.quality = quality;
        query.output_width = output_width;
        query.output_height = output_height;
        NgxRuntime* ngx = ngx_ ? ngx_() : nullptr;
        if (ngx == nullptr)
        {
            query.result.reason = CAELIX_UPSCALER_NO_NGX;
        }
        else
        {
            ngx->Query(static_cast<UpscalerKind>(upscaler), quality, output_width, output_height, query.result);
        }
        query.valid = true;
        if (query.result.supported == 0)
        {
            // Logged once per query key: the result is cached until the options change.
            CAELIX_LOGE("caelix_renderer_render: upscaler %u at quality %u for a %ux%u colour target is not "
                        "available (CaelixUpscalerUnsupportedReason %u; check caelix_upscaler_query first)",
                        upscaler, quality, output_width, output_height, query.result.reason);
        }
    }
    if (query.result.supported == 0)
    {
        return CAELIX_ERROR_UNSUPPORTED;
    }
    render_width = query.result.render_width;
    render_height = query.result.render_height;
    return CAELIX_OK;
#else
    static_cast<void>(quality);
    static_cast<void>(output_width);
    static_cast<void>(output_height);
    render_width = 0;
    render_height = 0;
    if (!upscale_unsupported_logged_)
    {
        CAELIX_LOGE("caelix_renderer_render: upscaler %u requested, but the library was built without "
                    "upscalers (CAELIX_DLSS=OFF)",
                    upscaler);
        upscale_unsupported_logged_ = true;
    }
    return CAELIX_ERROR_UNSUPPORTED;
#endif
}

Upscaler* Renderer::EnsureUpscaler(const UpscalerDesc& desc, VkCommandBuffer cmd, uint64_t frame, bool& error)
{
    if (has_upscaler_desc_ && upscaler_desc_ == desc)
    {
        return upscaler_failed_ ? nullptr : upscaler_.get();
    }
    // New options: the old upscaler and its output wait until their last frame is safe.
    if (upscaler_ != nullptr || upscale_out_.valid())
    {
        RetiredUpscaler retired;
        retired.frame = upscaler_last_frame_;
        retired.upscaler = std::move(upscaler_);
        retired.output = std::move(upscale_out_);
        retired_upscalers_.push_back(std::move(retired));
    }
    has_upscaler_desc_ = false;
    // On a throw nothing is kept, and the next upscaled render tries again. TRANSFER_DST: DLSS
    // clears its output with vkCmdClearColorImage ("fast UAV clear").
    upscale_out_ = GpuImage(device_.allocator(), device_.device(), desc.output_width, desc.output_height,
                            VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            "upscale_out");
    upscale_out_needs_init_ = true;
    upscaler_desc_ = desc;
    has_upscaler_desc_ = true;
    upscaler_failed_ = false;
    upscaler_new_ = true;
    upscaler_last_frame_ = frame;
    upscale_last_time_ = {};
#if CAELIX_DLSS
    NgxRuntime* ngx = ngx_ ? ngx_() : nullptr;
    if (ngx == nullptr)
    {
        // QueryUpscaler found NGX before; it cannot be missing now, but stay safe.
        CAELIX_LOGE("caelix_renderer_render: NGX is not available; the renderer falls back to the nearest "
                    "upscale");
        upscaler_failed_ = true;
        error = true;
        return nullptr;
    }
    try
    {
        upscaler_ = std::make_unique<DlssUpscaler>(*ngx, desc, cmd);
    }
    catch (const NgxError& e)
    {
        CAELIX_LOGE("caelix_renderer_render: %s; the renderer falls back to the nearest upscale until the "
                    "upscaler options change",
                    e.what());
        upscaler_failed_ = true;
        error = true;
        return nullptr;
    }
    catch (...)
    {
        // A Vulkan failure (the exposure image): nothing is kept, the next render tries again.
        has_upscaler_desc_ = false;
        throw;
    }
    CAELIX_LOGI("renderer: upscaler %u at quality %u: %ux%u -> %ux%u", static_cast<unsigned>(desc.kind),
                desc.quality, desc.render_width, desc.render_height, desc.output_width, desc.output_height);
    return upscaler_.get();
#else
    static_cast<void>(cmd);
    upscaler_failed_ = true;
    error = true;
    return nullptr;
#endif
}

void Renderer::GetUpscalerInfo(CaelixUpscalerInfo& out) const
{
    out = CaelixUpscalerInfo{};
    if (!has_upscaler_desc_)
    {
        return;
    }
    out.upscaler = static_cast<uint32_t>(upscaler_desc_.kind);
    out.quality = upscaler_desc_.quality;
    out.preset = upscaler_desc_.preset;
    out.render_width = upscaler_desc_.render_width;
    out.render_height = upscaler_desc_.render_height;
    out.output_width = upscaler_desc_.output_width;
    out.output_height = upscaler_desc_.output_height;
    out.failed = upscaler_failed_ || (upscaler_ != nullptr && upscaler_->failed()) ? 1u : 0u;
    out.depth_inverted = upscaler_desc_.depth_inverted ? 1u : 0u;
}

}  // namespace caelix
