"""Qt-free adapter for the declared mixed STEM result and existing image views."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np

from temsim.detector.stem_signal import CollectionAngle, DetectorSignal, StemScanResult
from temsim.immutable_json import thaw_json

MIXED_STEM_MODEL = "near_specimen_mixed_coherent_cuda"
ELEMENTARY_CHARGE_C = 1.602176634e-19


def _readonly(values):
    array = np.ascontiguousarray(values, dtype=float)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def adapt_mixed_stem(acquisition, request=None):
    """Preserve emitted-source fractions; ideal ADF uses the existing third panel."""
    from temsim.immutable_json import json_digest
    executed = executed_mixed_request(acquisition)
    request = executed if request is None else request.validate()
    if json_digest(asdict(request)) != json_digest(asdict(executed)):
        raise ValueError("Mixed STEM display request differs from the executed numerical metadata")
    state, scan = acquisition.state_snapshot, acquisition.scan
    saved = bool(getattr(acquisition, "read_only", False))
    x, y = np.asarray(scan.scan_x_nm, float), np.asarray(scan.scan_y_nm, float)
    selected_shape = (int(state.ac_deflector.scan_lines), int(state.ac_deflector.scan_pixels_x))
    if x.shape != y.shape or x.shape != selected_shape or x.ndim != 2 or not x.size:
        raise ValueError("Mixed STEM must return the complete selected two-dimensional raster")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Mixed STEM returned non-finite scan coordinates")
    checkpoint = scan.source_checkpoint
    from temsim.mixed_stem_artifact import SavedSourceSummary
    if saved != isinstance(checkpoint, SavedSourceSummary) or (saved
            and getattr(acquisition, "particle_result", None) is not None):
        raise ValueError("A saved mixed view requires its explicit read-only source summary")
    source_current = float(checkpoint.reference_current_a)
    if not np.isfinite(source_current) or source_current <= 0:
        raise ValueError("Mixed STEM requires its actual emitted-current reference")
    record = thaw_json(scan.record)
    original_count = int(checkpoint.record.get("original_positive_count", 0))
    if original_count < 1000:
        raise ValueError("Mixed STEM must retain at least 1000 original positive-current particles")
    fractions, current_pa, signals, windows = {}, {}, {}, {}
    for detector in request.detectors:
        if detector.key not in {"bf", "adf"}:
            raise ValueError("The mixed STEM display supports explicit BF and ADF ideal angle windows")
        key = "haadf" if detector.key == "adf" else "bf"
        values = np.asarray(scan.fractions[detector.key], dtype=float)
        current = np.asarray(scan.current_a[detector.key], dtype=float)
        if (values.shape != x.shape or current.shape != x.shape or not np.isfinite(values).all()
                or not np.isfinite(current).all() or np.any(values < 0.) or np.any(values > 1.+1e-10)):
            raise ValueError("Mixed STEM detector values must match the complete raster and emitted reference")
        if not np.allclose(current, source_current*values, rtol=1e-12, atol=0.):
            raise ValueError("Mixed STEM current and original emitted-source fractions disagree")
        fractions[key], current_pa[key] = _readonly(values), _readonly(current*1e12)
        mean = float(values.mean())
        label = "ADF" if key == "haadf" else "BF"
        signals[key] = DetectorSignal(key, label+" ideal exit-angle window", mean,
            original_count*mean, source_current*1e12*mean, source_current*mean/ELEMENTARY_CHARGE_C,
            CollectionAngle(detector.inner_mrad, detector.outer_mrad,
                (detector.inner_mrad, detector.inner_mrad), (detector.outer_mrad, detector.outer_mrad), False))
        windows[key] = [float(detector.inner_mrad), float(detector.outer_mrad)]
    detected = sum(fractions.values(), np.zeros_like(x))
    if np.any(detected > 1.+1e-10):
        raise ValueError("Mixed STEM ideal detector fractions exceed the emitted reference")
    dwell = float(state.ac_deflector.scan_frame_period_s)/x.size
    if not np.isfinite(dwell) or dwell <= 0:
        raise ValueError("Mixed STEM requires a positive selected dwell time")
    expected = {key: _readonly(value*1e-12*dwell/ELEMENTARY_CHARGE_C) for key,value in current_pa.items()}
    poisson = None
    if not saved and bool(getattr(state.sample, "stem_poisson_enabled", False)):
        seed = int(getattr(state.sample, "stem_poisson_seed", 0))
        if seed < 0:
            raise ValueError("Mixed STEM exposure seed must be nonnegative")
        rng = np.random.default_rng(seed)
        poisson = {key: rng.poisson(values) for key,values in expected.items()}
    band_loss = np.asarray(scan.fractions.get("numerical_band_loss", np.zeros_like(x)), float)
    if band_loss.shape == ():
        band_loss = np.full_like(x,float(band_loss))
    if band_loss.shape != x.shape or not np.isfinite(band_loss).all() or np.any(band_loss < 0.):
        raise ValueError("Mixed STEM numerical-band loss must match the raster")
    metrics = dict(record)
    metrics.update(model=MIXED_STEM_MODEL, mixed_scan_record=record,
        mixed_scan_request=asdict(request), ideal_exit_angle_detectors_mrad=windows,
        detector_model="Ideal specimen-exit angular windows; not installed detector stations",
        model_limitation="Rigid specimen-plane translation of the captured mixed probe; scan-coil waveforms and downstream detector conjugacy are not retraced.",
        source_current_pa=source_current*1e12, signal_fraction_reference="original emitted source current",
        original_positive_particles=original_count,
        incident_sample_fraction=float(checkpoint.record.get("source_fraction", checkpoint.beam.total_weight)),
        source_checkpoint_digest=checkpoint.digest, dwell_time_s=dwell,
        scan_frame_period_s=float(state.ac_deflector.scan_frame_period_s),
        scan_pixel_size_nm=float(state.ac_deflector.scan_pixel_size_nm),
        scan_field_of_view_x_nm=float(state.ac_deflector.scan_pixels_x)*float(state.ac_deflector.scan_pixel_size_nm),
        scan_field_of_view_y_nm=float(state.ac_deflector.scan_lines)*float(state.ac_deflector.scan_pixel_size_nm),
        source_electrons_per_pixel=source_current*dwell/ELEMENTARY_CHARGE_C,
        array_axis_order="y,x", laboratory_orientation="+X right; +Y up; electron beam +Z",
        poisson_noise_enabled=poisson is not None,
        uncollected_scope="All emitted-source probability outside the two ideal windows; includes upstream loss and numerical truncation, not an absorption measurement")
    metrics["read_only_saved_acquisition"] = saved
    if saved:
        original_inputs = getattr(acquisition, "input_snapshot", None)
        if original_inputs is not None and original_inputs.digest != record.get("input_digest"):
            raise ValueError("Saved mixed display metadata does not match its original input record")
        compatibility = getattr(acquisition, "input_compatibility", None)
        if compatibility is not None:
            compatibility = thaw_json(compatibility)
            if compatibility.get("physical_inputs_unchanged") is not True:
                raise ValueError("Saved mixed display migration changed physical inputs")
        metrics["saved_input_digest"] = record.get("input_digest")
        metrics["saved_input_compatibility"] = compatibility
    frame = StemScanResult(_readonly(x*1e-3), _readonly(y*1e-3), fractions, signals,
        metrics=metrics, current_pa=current_pa, expected_electrons=expected, poisson_counts=poisson,
        dwell_time_s=dwell, uncollected_fraction=_readonly(np.maximum(1.-detected,0.)),
        truncated_fraction=_readonly(band_loss))
    pattern = getattr(scan,"diffraction_probability",None)
    diffraction = None
    if pattern is not None:
        pattern = np.asarray(pattern,float)
        fx,fy = np.asarray(scan.qx_inv_angstrom,float),np.asarray(scan.qy_inv_angstrom,float)
        if fx.ndim==fy.ndim==1:
            fx,fy=np.meshgrid(fx,fy)
        if (pattern.ndim!=2 or min(pattern.shape)<2 or pattern.shape!=fx.shape or pattern.shape!=fy.shape
                or not np.isfinite(pattern).all() or np.any(pattern<0.)
                or not np.isfinite(fx).all() or not np.isfinite(fy).all()):
            raise ValueError("Mixed STEM exit diffraction must have matching finite frequency coordinates")
        dx,dy=np.diff(fx[0]),np.diff(fy[:,0])
        if (np.any(dx<=0) or np.any(dy<=0)
                or not np.allclose(dx,dx[0],rtol=1e-10,atol=0.)
                or not np.allclose(dy,dy[0],rtol=1e-10,atol=0.)
                or not np.array_equal(fx,np.broadcast_to(fx[0],fx.shape))
                or not np.array_equal(fy,np.broadcast_to(fy[:,0,None],fy.shape))):
            raise ValueError("Mixed STEM diffraction display needs increasing uniform Cartesian frequency axes")
        diffraction=SimpleNamespace(probability=_readonly(pattern),frequency_x_inv_angstrom=_readonly(fx),
            frequency_y_inv_angstrom=_readonly(fy),position_nm=record.get("diffraction_position_nm"),
            reference="Probability per original emitted electron per reciprocal-space bin",
            read_only_saved_acquisition=saved)
        metrics["mixed_exit_diffraction"] = vars(diffraction).copy()
    return frame,diffraction


def executed_mixed_request(acquisition):
    """Restore executed numerical metadata; never substitute current UI drafts."""
    from temsim.physics.mixed_stem_scan import MixedStemScanRequest
    from temsim.physics.particle_wave_source import MixedSourceNumerics
    from temsim.physics.stem_wave_imaging import AngularDetector
    record = thaw_json(acquisition.scan.record)
    data = dict(record.get("request", {}))
    if not data:
        raise ValueError("The mixed acquisition has no executed request metadata")
    data["source"] = MixedSourceNumerics(**data["source"])
    data["detectors"] = tuple(AngularDetector(**row) for row in data["detectors"])
    return MixedStemScanRequest(**data).validate()


def wrap_mixed_stem_result(acquisition, frame):
    """Create a display/archive owner, never insert it into particle caches."""
    if getattr(acquisition, "particle_result", None) is None:
        raise ValueError("Saved mixed acquisitions are read-only images, without a reconstructible ray history")
    from temsim.calculation_cache import state_model_signature
    from temsim.immutable_json import json_digest
    signatures = dict(acquisition.particle_result.signatures or {})
    identity = json_digest({"schema": "mixed-stem-display-v1",
        "source_checkpoint": frame.metrics["source_checkpoint_digest"],
        "request": frame.metrics["mixed_scan_request"]})
    signatures.update(stem=identity, request=identity, workflow="stem")
    performance = dict(acquisition.particle_result.performance or {})
    performance.update(workflow="stem", mixed_stem_request_identity=identity)
    return replace(acquisition.particle_result, state_snapshot=acquisition.state_snapshot,
        stem_scan=frame, workflow="stem", signatures=signatures,
        model_signature=state_model_signature(acquisition.state_snapshot),
        calculated_products=frozenset(("stem",)), reused_products=frozenset(),
        cache_hit=False, performance=performance)
