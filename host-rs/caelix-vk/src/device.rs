//! Instance creation, physical device selection and logical device creation.

use std::ffi::{c_char, CStr, CString};

use ash::vk;
use raw_window_handle::RawDisplayHandle;

use crate::{debug, vk_error, DeviceRequirements, Error, Features};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

pub(crate) type DebugMessenger = (ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT);

pub(crate) fn create_instance(
    entry: &ash::Entry,
    display_handle: RawDisplayHandle,
    api_version: u32,
    enable_validation: bool,
) -> Result<(ash::Instance, Option<DebugMessenger>), Error> {
    let mut extensions: Vec<*const c_char> = ash_window::enumerate_required_extensions(display_handle)
        .map_err(vk_error("ash_window::enumerate_required_extensions"))?
        .to_vec();

    let mut layers: Vec<*const c_char> = Vec::new();
    let mut validation = false;
    if enable_validation {
        // SAFETY: plain enumeration through the loader.
        let available = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(vk_error("vkEnumerateInstanceLayerProperties"))?;
        if available
            .iter()
            .any(|layer| layer.layer_name_as_c_str().ok() == Some(VALIDATION_LAYER))
        {
            layers.push(VALIDATION_LAYER.as_ptr());
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            validation = true;
        } else {
            log::warn!("validation requested, but VK_LAYER_KHRONOS_validation is not installed");
        }
    }

    let app_name = c"Caelix Renderer";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .application_version(1)
        .engine_name(app_name)
        .engine_version(1)
        .api_version(api_version);
    // Chained into the create info so the messenger also reports messages from
    // vkCreateInstance and vkDestroyInstance.
    let mut messenger_info = debug::messenger_create_info();
    let mut create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions);
    if validation {
        create_info = create_info.push_next(&mut messenger_info);
    }
    // SAFETY: every pointer in create_info outlives the call.
    let instance = unsafe { entry.create_instance(&create_info, None) }
        .map_err(vk_error("vkCreateInstance"))?;

    let messenger = if validation {
        let loader = ash::ext::debug_utils::Instance::new(entry, &instance);
        // SAFETY: the instance has VK_EXT_debug_utils enabled.
        let messenger =
            unsafe { loader.create_debug_utils_messenger(&debug::messenger_create_info(), None) }
                .map_err(vk_error("vkCreateDebugUtilsMessengerEXT"))?;
        log::info!("Vulkan validation layer enabled");
        Some((loader, messenger))
    } else {
        None
    };
    Ok((instance, messenger))
}

pub(crate) struct SelectedDevice {
    pub physical_device: vk::PhysicalDevice,
    pub queue_family: u32,
}

/// The features this crate needs for itself, on top of the library's.
fn host_features(requested: &Features) -> Features {
    Features { synchronization2: true, ..*requested }
}

pub(crate) fn select_physical_device(
    instance: &ash::Instance,
    surface_fn: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    requirements: &DeviceRequirements,
    api_version: u32,
) -> Result<SelectedDevice, Error> {
    // SAFETY: plain enumeration on a live instance.
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(vk_error("vkEnumeratePhysicalDevices"))?;
    let features = host_features(&requirements.features);

    let mut first_suitable: Option<SelectedDevice> = None;
    let mut rejections: Vec<String> = Vec::new();
    for physical_device in physical_devices {
        // SAFETY: the physical device comes from this instance.
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let name = properties
            .device_name_as_c_str()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        match check_device(instance, surface_fn, surface, physical_device, requirements, &features, api_version) {
            Ok(queue_family) => {
                let selected = SelectedDevice { physical_device, queue_family };
                if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                    log::info!("selected Vulkan device: {name} (discrete)");
                    return Ok(selected);
                }
                if first_suitable.is_none() {
                    first_suitable = Some(selected);
                }
            }
            Err(reason) => {
                log::debug!("Vulkan device {name} rejected: {reason}");
                rejections.push(format!("{name}: {reason}"));
            }
        }
    }
    match first_suitable {
        Some(selected) => {
            log::info!("selected Vulkan device: no suitable discrete GPU, using the first suitable device");
            Ok(selected)
        }
        None => Err(Error::NoSuitableDevice(if rejections.is_empty() {
            "no Vulkan device found".into()
        } else {
            rejections.join("; ")
        })),
    }
}

/// Returns the graphics + compute + present queue family, or why the device is unsuitable.
fn check_device(
    instance: &ash::Instance,
    surface_fn: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    requirements: &DeviceRequirements,
    features: &Features,
    api_version: u32,
) -> Result<u32, String> {
    // SAFETY (whole function): queries on a physical device of this instance.
    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    if properties.api_version < api_version {
        return Err(format!(
            "Vulkan {}.{} < required {}.{}",
            vk::api_version_major(properties.api_version),
            vk::api_version_minor(properties.api_version),
            vk::api_version_major(api_version),
            vk::api_version_minor(api_version)
        ));
    }

    let available = unsafe { instance.enumerate_device_extension_properties(physical_device) }
        .map_err(|e| format!("vkEnumerateDeviceExtensionProperties: {e:?}"))?;
    let has_extension = |name: &CStr| {
        available
            .iter()
            .any(|e| e.extension_name_as_c_str().ok() == Some(name))
    };
    for name in device_extensions(requirements) {
        if !has_extension(&name) {
            return Err(format!("missing extension {}", name.to_string_lossy()));
        }
    }

    // The extension feature structs are only chained when the extension exists.
    let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut acceleration_structure = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
    let mut ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default();
    let has_as = has_extension(ash::khr::acceleration_structure::NAME);
    let has_rq = has_extension(ash::khr::ray_query::NAME);
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut v12)
        .push_next(&mut v13);
    if has_as {
        features2 = features2.push_next(&mut acceleration_structure);
    }
    if has_rq {
        features2 = features2.push_next(&mut ray_query);
    }
    unsafe { instance.get_physical_device_features2(physical_device, &mut features2) };
    let core = features2.features;

    let checks: [(bool, vk::Bool32, &str); 12] = [
        (features.buffer_device_address, v12.buffer_device_address, "bufferDeviceAddress"),
        (features.descriptor_indexing, v12.runtime_descriptor_array, "runtimeDescriptorArray"),
        (
            features.descriptor_indexing,
            v12.shader_storage_buffer_array_non_uniform_indexing,
            "shaderStorageBufferArrayNonUniformIndexing",
        ),
        (features.descriptor_indexing, v12.descriptor_binding_partially_bound, "descriptorBindingPartiallyBound"),
        (
            features.descriptor_indexing,
            v12.descriptor_binding_variable_descriptor_count,
            "descriptorBindingVariableDescriptorCount",
        ),
        (features.synchronization2, v13.synchronization2, "synchronization2"),
        (features.dynamic_rendering, v13.dynamic_rendering, "dynamicRendering"),
        (features.timeline_semaphore, v12.timeline_semaphore, "timelineSemaphore"),
        (
            features.acceleration_structure,
            if has_as { acceleration_structure.acceleration_structure } else { vk::FALSE },
            "accelerationStructure",
        ),
        (features.ray_query, if has_rq { ray_query.ray_query } else { vk::FALSE }, "rayQuery"),
        (features.shader_int64, core.shader_int64, "shaderInt64"),
        (features.scalar_block_layout, v12.scalar_block_layout, "scalarBlockLayout"),
    ];
    for (required, supported, name) in checks {
        if required && supported != vk::TRUE {
            return Err(format!("missing feature {name}"));
        }
    }

    let families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    for (index, family) in families.iter().enumerate() {
        let index = index as u32;
        if !family
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
        {
            continue;
        }
        let present = unsafe {
            surface_fn.get_physical_device_surface_support(physical_device, index, surface)
        }
        .unwrap_or(false);
        if present {
            return Ok(index);
        }
    }
    Err("no queue family with graphics + compute + present".into())
}

/// The library's extensions plus VK_KHR_swapchain, without duplicates.
fn device_extensions(requirements: &DeviceRequirements) -> Vec<CString> {
    let mut names: Vec<CString> = requirements.device_extensions.clone();
    let swapchain = CString::from(ash::khr::swapchain::NAME);
    if !names.contains(&swapchain) {
        names.push(swapchain);
    }
    names
}

pub(crate) fn create_device(
    instance: &ash::Instance,
    selected: &SelectedDevice,
    requirements: &DeviceRequirements,
) -> Result<(ash::Device, vk::Queue), Error> {
    let features = host_features(&requirements.features);
    let extension_names = device_extensions(requirements);
    let extension_ptrs: Vec<*const c_char> = extension_names.iter().map(|n| n.as_ptr()).collect();

    let priorities = [1.0f32];
    let queue_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(selected.queue_family)
        .queue_priorities(&priorities)];

    let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(features.buffer_device_address)
        .runtime_descriptor_array(features.descriptor_indexing)
        .shader_storage_buffer_array_non_uniform_indexing(features.descriptor_indexing)
        .descriptor_binding_partially_bound(features.descriptor_indexing)
        .descriptor_binding_variable_descriptor_count(features.descriptor_indexing)
        .timeline_semaphore(features.timeline_semaphore)
        .scalar_block_layout(features.scalar_block_layout);
    let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(features.synchronization2)
        .dynamic_rendering(features.dynamic_rendering);
    let mut acceleration_structure =
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
    let mut ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::default().ray_query(true);
    let core = vk::PhysicalDeviceFeatures::default().shader_int64(features.shader_int64);
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .features(core)
        .push_next(&mut v12)
        .push_next(&mut v13);
    if features.acceleration_structure {
        features2 = features2.push_next(&mut acceleration_structure);
    }
    if features.ray_query {
        features2 = features2.push_next(&mut ray_query);
    }

    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extension_ptrs)
        .push_next(&mut features2);
    // SAFETY: every pointer in create_info outlives the call; support was checked
    // in select_physical_device.
    let device = unsafe { instance.create_device(selected.physical_device, &create_info, None) }
        .map_err(vk_error("vkCreateDevice"))?;
    // SAFETY: queue 0 of the family was requested above.
    let queue = unsafe { device.get_device_queue(selected.queue_family, 0) };
    log::debug!(
        "device created: extensions [{}], features {features:?}",
        extension_names
            .iter()
            .map(|n| n.to_string_lossy())
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok((device, queue))
}
