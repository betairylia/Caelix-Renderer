//! Debug-utils messenger that forwards validation and loader messages to `log`
//! with target `"vulkan"`.

use std::ffi::c_void;

use ash::vk;

pub(crate) fn messenger_create_info() -> vk::DebugUtilsMessengerCreateInfoEXT<'static> {
    vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback))
}

/// VERBOSE/INFO -> debug, WARNING -> warn, ERROR -> error.
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> vk::Bool32 {
    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::Level::Error
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::Level::Warn
    } else {
        log::Level::Debug
    };
    if !log::log_enabled!(target: "vulkan", level) {
        return vk::FALSE;
    }
    // SAFETY: Vulkan passes a valid callback data struct whose strings live for
    // the duration of the call.
    let message = unsafe { data.as_ref().and_then(|d| d.message_as_c_str()) }
        .map(|m| m.to_string_lossy())
        .unwrap_or_default();
    log::log!(target: "vulkan", level, "[{message_type:?}] {message}");
    vk::FALSE
}
