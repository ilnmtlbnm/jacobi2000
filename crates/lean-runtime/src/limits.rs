//! Size checks against the device's limits, made before a buffer or a
//! dispatch is created so an oversized request is an error, not a device
//! loss or a validation panic.

use crate::Error;

/// The largest buffer that can also be bound whole as storage.
pub fn max_binding_bytes(l: &wgpu::Limits) -> u64 {
    l.max_buffer_size.min(l.max_storage_buffer_binding_size as u64)
}

/// Ok if a `bytes`-long storage buffer can be created and bound whole.
pub fn check_binding(l: &wgpu::Limits, what: &str, bytes: u64) -> Result<(), Error> {
    let have = max_binding_bytes(l);
    if bytes > have {
        return Err(Error::Limit { what: format!("{what} (bytes in one storage binding)"), need: bytes, have });
    }
    Ok(())
}

/// Ok if a dispatch of `groups` workgroups fits in every dimension.
pub fn check_workgroups(l: &wgpu::Limits, what: &str, groups: (u32, u32, u32)) -> Result<(), Error> {
    let need = groups.0.max(groups.1).max(groups.2);
    let have = l.max_compute_workgroups_per_dimension;
    if need > have {
        return Err(Error::Limit { what: format!("{what} (workgroups in one dimension)"), need: need as u64, have: have as u64 });
    }
    Ok(())
}

/// Ok if the device runs workgroups of `invocations` threads with
/// `workgroup_storage` bytes of workgroup memory.
pub fn require_compute(l: &wgpu::Limits, invocations: u32, workgroup_storage: u32) -> Result<(), Error> {
    if l.max_compute_invocations_per_workgroup < invocations {
        return Err(Error::Limit { what: "invocations per workgroup".into(), need: invocations as u64, have: l.max_compute_invocations_per_workgroup as u64 });
    }
    if l.max_compute_workgroup_storage_size < workgroup_storage {
        return Err(Error::Limit { what: "workgroup storage bytes".into(), need: workgroup_storage as u64, have: l.max_compute_workgroup_storage_size as u64 });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks() {
        let l = wgpu::Limits { max_buffer_size: 1 << 30, max_storage_buffer_binding_size: 1 << 27, max_compute_workgroups_per_dimension: 65535, ..wgpu::Limits::downlevel_defaults() };
        assert_eq!(max_binding_bytes(&l), 1 << 27);
        assert!(check_binding(&l, "state", 1 << 27).is_ok());
        assert!(matches!(check_binding(&l, "state", (1 << 27) + 1), Err(Error::Limit { have: 134217728, .. })));
        assert!(check_workgroups(&l, "step", (65535, 1, 1)).is_ok());
        assert!(check_workgroups(&l, "step", (1, 65536, 1)).is_err());
        assert!(require_compute(&wgpu::Limits::default(), 256, 16384).is_ok());
        assert!(require_compute(&wgpu::Limits::default(), 1024, 16384).is_err());
    }
}
