"""Explicit wave admission boundary for newly installed ideal coil sources."""


def require_transverse_coil_wave_admission(snapshot):
    """Reject unqualified coil waves before allocation or native substitution.

    The closed-pack provider retains A and its derivatives for classical Lorentz
    execution. That does not establish its coherent residual bounds or CUDA
    wave implementation. Existing native providers keep their own admission.
    """
    from temsim.physics.transverse_coil_field import TransverseCoilPotential
    from temsim.physics.dynamic_transverse_coil import DynamicTransverseCoilPotential
    unsupported=(TransverseCoilPotential,DynamicTransverseCoilPotential)
    for row in snapshot.contributions:
        if type(row) in unsupported or type(getattr(row,'potential',None)) in unsupported:
            raise ValueError('Ideal transverse coil fields currently support classical particles only; '
                'coherent propagation is not qualified and native multipoles cannot be substituted')
