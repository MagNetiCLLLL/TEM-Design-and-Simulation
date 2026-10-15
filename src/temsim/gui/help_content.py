"""User-facing help topics, independent of widgets and numerical execution."""
from dataclasses import dataclass


@dataclass(frozen=True)
class HelpTopic:
    key: str
    title: str
    summary: str
    keywords: tuple[str, ...]
    html: str

HELP_TOPICS: tuple[HelpTopic, ...] = (
    HelpTopic(
        "quick_start",
        "Quick start",
        "Build an instrument, choose an operating preset and calculate your first beam.",
        ("start", "beginner", "workflow", "preview", "calculate", "offline"),
        """
        <p>Explore electron optics and simulated microscopy signals in an offline instrument.
        Start with the beam in vacuum, then add specimen and detector calculations.</p>
        <h3>Try it</h3>
        <ol>
          <li>In <b>Instrument configuration</b>, choose <b>Configure instrument…</b>,
          select the required units and click <b>Assemble</b>.</li>
          <li>Under <b>Optical operating preset</b>, choose illumination and projection,
          then click <b>Apply calculated lens preset</b>.</li>
          <li>Click <b>Update rays</b> for a fast preview. Inspect <b>Ray Diagram</b>
          and select an axial Z position.</li>
          <li>Set <b>High-accuracy rays</b>, <b>Step (mm)</b> and <b>Compute</b>,
          then click <b>Run high-accuracy once</b>.</li>
          <li>For material signals, configure <b>Sample</b> and use the calculation
          button on the desired specimen, image or spectrum page.</li>
        </ol>
        <h3>Read the result</h3>
        <p>Check the result status and captured settings before comparing plots.
        A completed calculation establishes a result for those settings and numerical
        choices; it does not certify a commercial microscope. The simulator does not
        control a real instrument.</p>
        <p>For coherent intensity and phase, use the separate
        <a href="help:coherent_beam">Electron beam workflow</a>.
        See also <a href="help:calculation">calculation controls</a>.</p>
        """,
    ),
    HelpTopic(
        "instrument",
        "Instrument configuration and components",
        "Choose installed units and edit the controls of individual components.",
        ("assembly", "FEG", "thermionic", "monochromator", "aperture", "lens",
         "deflector", "stigmator", "source", "CETCOR", "B-COR"),
        """
        <p><b>Where:</b> the instrument panel, <b>Configure instrument…</b>,
        and the <b>Optical</b>, <b>Mechanical</b> and <b>Assembly</b> component lists.</p>
        <h3>Choose hardware</h3>
        <ol>
          <li>Open <b>Configure instrument…</b>. Select the electron source and
          optional monochromator, beam blanker, C3 lens, correctors and energy filter.</li>
          <li>Review the linked physical layout and any compatibility message.</li>
          <li>Click <b>Assemble</b> to apply the selection.</li>
          <li>Select a component to inspect its available operating controls,
          position and geometry. Check the displayed units before editing.</li>
        </ol>
        <p>C1/C2, the objective, sample stage and recording hardware are part of the
        base assembly. Optional image-corrector choices include CETCOR and B-COR.
        Available controls depend on the installed hardware.</p>
        <h3>Read the result</h3>
        <p>Installed, enabled and inserted describe different hardware states.
        An aperture opening can remove electrons even when the drawing looks similar.
        Mechanical dimensions and effective optical field lengths are also separate
        quantities. After an edit, check whether displayed results belong to the
        previous settings and calculate again when needed.</p>
        <p><a href="help:layout">Physical Layout</a> explains geometry views;
        <a href="help:correctors">Hardware tuning</a> explains alignment controls.</p>
        """,
    ),
    HelpTopic(
        "calculation",
        "Operating modes and calculations",
        "Apply illumination and projection presets, choose accuracy and manage calculation scope.",
        ("Microprobe", "Nanoprobe", "Normal", "Large angle", "Image", "Diffraction",
         "CPU", "GPU", "live tuning", "cutoff", "cancel", "cache", "Simulation"),
        """
        <p><b>Where:</b> <b>Optical operating preset</b>, the top calculation toolbar,
        <b>Simulation</b>, and <b>Live tuning → Calculation</b>.</p>
        <ol>
          <li>Select <b>Probe / illumination</b>: Microprobe or Nanoprobe.
          For Nanoprobe, choose the available <b>Nanoprobe range</b>.</li>
          <li>Select <b>Projection</b>: Image or Diffraction. Click
          <b>Apply calculated lens preset</b> to change the optical controls.</li>
          <li>Use <b>Calculate setup</b> to review calculation settings.
          Use <b>Update rays</b> for a preview and <b>Run high-accuracy once</b>
          for a captured particle calculation. The latter is also available through
          <b>Simulation → Calculate classical rays</b>.</li>
          <li>Use the page's own Calculate button to acquire its image or spectrum.
          Simply opening a tab does not acquire a new result.</li>
        </ol>
        <p>Normal and Large angle select different optical branches; their labels alone
        do not specify the achieved convergence angle. Read the preset status and
        measured beam result. Illumination and downstream projection have separate roles.</p>
        <p><b>Compute</b> applies across pages. Auto tries supported GPU work and can
        fall back to CPU; Require GPU reports a failure when the required backend is
        unavailable. Read the completed result's actual backend.</p>
        <p>Live tuning can capture settings, vary selected ranges and reuse compatible
        upstream calculations. Finish a comparison with appropriate ray counts and
        step-size checks. <b>Cancel calculations</b> requests cancellation while keeping
        previous complete results. See <a href="help:working_points">saved results</a>.</p>
        """,
    ),
    HelpTopic(
        "rays",
        "Ray Diagram and transverse views",
        "Follow rays from the source, inspect a selected Z and search conjugate planes.",
        ("transverse", "XY", "X-Y", "origin", "zero", "Fit beam", "source position",
         "source angle", "colour", "conjugate", "magnetic", "virtual electrons"),
        """
        <p><b>Where:</b> <b>Ray Diagram</b>, its transverse <b>Classical rays</b>
        view, and <b>Conjugate planes</b>.</p>
        <ol>
          <li>Calculate rays, then select a component or axial Z to inspect its
          cross-section.</li>
          <li>Compare the source plot with the selected-plane plot. Source colours
          track the original emission position or direction; read the colour legend.</li>
          <li>Use <b>Fit beam</b> when you want to reset the classical transverse
          scale. <b>Plot sizes…</b> changes the panel dimensions.</li>
          <li>Open <b>Conjugate planes</b> and click <b>Find conjugate planes</b>
          to pin the current reference. Select a result to inspect its Z.
          Use <b>Use selected Z</b> before starting a search from another reference.</li>
        </ol>
        <h3>Read the result</h3>
        <p>The classical transverse view keeps physical (0, 0) at the plot centre.
        Changing Z or view angle preserves this reference; Fit beam changes the scale
        without centring on a displaced beam. A displaced spot can therefore reveal
        steering rather than a display offset.</p>
        <p>Ray colours and path lines describe the displayed particle calculation.
        Virtual electrons are separately adjustable diagnostic paths. Neither is a
        coherent phase measurement. Conjugate-plane results describe first-order
        imaging geometry, including magnification and rotation; they do not establish
        transmission or diffraction intensity.</p>
        <p>See <a href="help:coherent_beam">Electron beam</a> for wave observations.</p>
        """,
    ),
    HelpTopic(
        "coherent_beam",
        "Electron beam and phase plates",
        "Calculate a captured coherent Tip state and inspect intensity, phase and probability flow.",
        ("coherence", "Apply tip parameters", "Calculate beam", "Beam observation",
         "phase plate", "wavefront", "tilt", "defocus", "sampling", "previous calculation",
         "mode", "Z", "arrivals", "piston", "reference"),
        """
        <p><b>Where:</b> <b>Electron beam</b>; its observation is also available under
        <b>Ray Diagram → Beam observation</b>.</p>
        <ol>
          <li>Review <b>Tip parameters</b>, choose a supported emission/coherence
          model and click <b>Apply tip parameters</b>. This applies the source edit
          without starting propagation.</li>
          <li>Select the current Tip, a saved emission state or an intensity overlay.
          Set the observation Z, then click <b>Calculate beam</b>.</li>
          <li>Use <b>Observe</b> to select Intensity, Electron arrivals, Beam current,
          Phase plate, Probability flow, Angular distribution or Interactions.</li>
          <li>For Phase plate, select one coherent mode. For a state overlay,
          first choose the member with <b>Inspect state</b>.</li>
        </ol>
        <h3>Read the result</h3>
        <p><b>Phase plate</b> shows the actual single-mode transverse wavefront,
        wrapped to [-π, π] radians. Tilt, defocus, aberrations and the inherited
        constant phase remain. It is a wave observation, not an inserted hardware
        phase plate. Independent modes add intensities; they have no single mixture phase.</p>
        <p>Changing supported Z requests an exact plane from the captured calculation.
        Cached compatible planes can be reused. This is not interpolation between
        pictures. Arbitrary planes inside the gun or material may be unsupported.</p>
        <p><b>Previous calculation</b> means the display belongs to earlier inputs.
        After edits, wait for a new completed result or use Calculate beam.
        Failure or cancellation can leave the previous complete observation visible.</p>
        <p>Wrapped colours alone do not establish adequate phase sampling.
        Fine fringes can remain unresolved even when an intensity or phase image is
        displayed. Check sampling limits and numerical convergence; the default
        Tip-to-image chain is not fully qualified.</p>
        <p>Electron arrivals are a simulated exposure of the calculated probabilities.
        Probability-flow arrows show local current directions, not measured trajectories.</p>
        <h3>Compare with an ideal lens reference</h3>
        <p>Choose <b>Wavefront comparison</b>, then <b>Compute reference</b> for an
        additional calculation. The reference retains focusing and first-order
        transport while removing the supported lens-aberration terms. Compare the
        matching mode at the same Z. <b>Remove constant phase offset</b> changes
        the difference view only. This comparison does not certify removal of
        energy-spread effects and does not include longitudinal/time phase.</p>
        """,
    ),
    HelpTopic(
        "correctors",
        "Hardware tuning and corrector channels",
        "Measure beam changes and apply local A1/A2 adjustments through physical simulator controls.",
        ("alignment", "Probe corrector", "A1", "A2", "coarse", "fine", "normal", "skew",
         "hexapole", "thickness", "field length", "spherical", "Cs", "baseline", "calibration"),
        """
        <p><b>Where:</b> <b>Hardware tuning</b>. Choose an alignment from the left
        list; the Corrector group includes <b>Hexapole probe corrector</b> and
        <b>A1 / A2 coarse and fine</b>.</p>
        <ol>
          <li>For ordinary tuning, select the hardware task and edit its available
          drives. Inspect <b>Executed beam feedback (read only)</b> at the desired Z.</li>
          <li>Use <b>Use displayed readout as baseline</b> to compare later
          completed beam measurements with that readout.</li>
          <li>For an aberration channel, choose the system and A1 or A2 coarse/fine
          channel, set <b>Calibration pupil</b> and <b>Maximum axial step</b>,
          then click <b>Calculate response</b>.</li>
          <li>Enter the requested signed normal/skew coefficient changes.
          Click <b>Calculate and apply change</b>, then inspect the measured response
          and calculate the beam observation required for your comparison.</li>
        </ol>
        <h3>Read the result</h3>
        <p>A1 describes twofold astigmatism; A2 describes threefold astigmatism.
        Coarse and fine channels use different physical drive groups.
        Their response is calibrated locally from the current simulator settings.
        It is not a manufacturer's channel map or a hardware calibration.
        Changed optics require a new response calculation.</p>
        <p>The diagnostic pupil may differ from the actual emitted beam.
        A successful local coefficient change does not establish correction across
        the full beam or a validated coherent probe.</p>
        <h3>Finite hexapole length</h3>
        <p>Effective field length and mechanical body length have different meanings.
        Changing effective length alters the distributed field action and can change
        spherical aberration as well as A2. The response also depends on strength,
        orientation and the surrounding relay optics. A field-strength setting is
        not itself a specimen-plane Cs value.</p>
        <p><a href="help:validation">Model validation</a> explains residual and
        convergence checks.</p>
        """,
    ),
    HelpTopic(
        "sample",
        "Sample and specimen interactions",
        "Add a finite specimen, import crystal data and calculate its interaction signals.",
        ("vacuum", "CIF", "MCIF", "crystal", "orientation", "zone axis", "tilt",
         "thickness", "inelastic", "material", "specimen"),
        """
        <p><b>Where:</b> <b>Sample</b>. A new sample starts as <b>Vacuum sample</b>.</p>
        <ol>
          <li>Use <b>Open CIF...</b> to import your crystal structure when material
          is required. Review the sample's dimensions and finite envelope.</li>
          <li>Set orientation using the available zone-axis or tilt controls.
          Use <b>Apply draft orientation</b> to publish a pending orientation edit.</li>
          <li>Prepare a compatible incident particle calculation with
          <b>Run high-accuracy once</b>.</li>
          <li>Click <b>Calculate sample</b> to update specimen interactions and
          downstream particle results.</li>
        </ol>
        <h3>Read the result</h3>
        <p>Dimensions, orientation and material determine which electrons meet the
        specimen and which interactions are modelled. A local structural preview
        helps inspect orientation and boundaries; it is not a calculated detector image.</p>
        <p>Importing a CIF does not select or enable coherent imaging. The simulator
        uses the supplied structure and model settings; it does not choose a reference
        material automatically. STEM frames, detector reception, EDS and energy-filter
        outputs have their own calculation controls.</p>
        <p>Continue with <a href="help:scanning">Scanning images and detectors</a>
        or <a href="help:spectroscopy">spectroscopy</a>.</p>
        """,
    ),
    HelpTopic(
        "scanning",
        "Scanning images and detectors",
        "Acquire a STEM frame, inspect detector reception and understand image-model limits.",
        ("STEM", "AC", "Descan", "raster", "HAADF", "BF", "DF", "camera",
         "screen", "4D-STEM", "projected atoms", "counts", "Poisson", "image"),
        """
        <p><b>Where:</b> <b>Scanning Image</b> for raster acquisition;
        <b>Illuminating Image → Camera / screen</b> for recording-plane reception.</p>
        <ol>
          <li>Prepare the current incident beam with <b>Run high-accuracy once</b>
          and configure the sample.</li>
          <li>Review AC scan and Descan settings, field of view, pixel sampling
          and detector geometry. Use the scan-calibration controls when required.</li>
          <li>Enable <b>Generate STEM detector images</b> and choose the
          <b>Particle model</b>.</li>
          <li>Click <b>Calculate STEM (single frame)</b>. Inspect Geometry and Images;
          use the displayed quantity and intensity range when comparing detectors.</li>
        </ol>
        <h3>Choose the interpretation</h3>
        <p><b>Projected atoms (fast approximation)</b> uses the imported atomic
        structure, calculated probe footprint and detector acceptance for approximate
        thin-sample contrast. Its BF/DF signals do not include Bragg interference
        or channeling. <b>Material particle paths</b> uses simulated material-scattering
        trajectories.</p>
        <p>Detector positions and inner/outer collection dimensions matter. A
        detector signal does not by itself prove the plane is a diffraction focus.
        Auto contrast changes display limits, not the stored signal.</p>
        <p>Expected counts and seeded Poisson counts describe different quantities.
        Changing their display selection does not acquire a new frame. The 4D-STEM
        area can inspect available diffraction cubes and integrate virtual annuli;
        this does not imply a newly validated full-column coherent acquisition.
        Respect any paused or unavailable model indication.</p>
        <p>For coherent source requirements, see <a href="help:coherent_beam">Electron beam</a>.</p>
        """,
    ),
    HelpTopic(
        "spectroscopy",
        "Energy Filter, EELS and EDS",
        "Calculate energy-loss and X-ray signals and inspect their associated geometry.",
        ("spectrum", "spectroscopy", "energy loss", "EELS", "EFTEM", "EDS", "X-ray",
         "energy filter", "detector", "interactions"),
        """
        <p><b>Where:</b> the <b>Energy Filter</b> and <b>EDS</b> pages.</p>
        <h3>Energy Filter</h3>
        <ol>
          <li>Assemble an energy filter and ensure the calculated electron path
          reaches its entrance.</li>
          <li>Review filter and acquisition settings, then click
          <b>Calculate energy filter</b>.</li>
          <li>Inspect <b>EELS spectrum</b> and <b>EFTEM image</b>.
          For structure, use <b>Physical Layout → Energy Filter</b>;
          for particle paths, use <b>Ray Diagram → Energy Filter rays</b>.</li>
        </ol>
        <h3>EDS</h3>
        <ol>
          <li>Configure material in Sample, prepare the incident beam and review
          the EDS acquisition and detector settings under <b>Parameters</b>.</li>
          <li>Click <b>Calculate EDS</b> and inspect <b>Spectrum</b>.</li>
          <li>For local trajectories, open <b>Interactions 3D</b> and use
          <b>Calculate detailed sample</b>. Detailed paths and the spectrum are
          separate calculations.</li>
        </ol>
        <h3>Read the result</h3>
        <p>EELS describes simulated electron energy losses; EDS describes simulated
        X-ray emission and detection. Their axes, collection geometry and acquisition
        assumptions differ. Read status and units, and recalculate after changing
        specimen or detector inputs. These are model outputs, not measured spectra
        or independently established chemical compositions.</p>
        """,
    ),
    HelpTopic(
        "working_points",
        "Working points and saved results",
        "Preserve settings and calculations, inspect compatibility and restore the intended data.",
        ("save", "load", "export", "import", "profile", "working point", "temresult",
         "temwp", "temdiag", "restore", "comparison", "archive"),
        """
        <p><b>Where:</b> <b>Live tuning → Working points</b>, the result toolbar,
        and <b>File → Settings</b>.</p>
        <ol>
          <li>Select a Working points record and read its source, assembly, mode,
          Z, compatibility and numerical-evidence columns.</li>
          <li>Use <b>Load retained data</b> to verify and load stored numbers.
          This does not run a calculation or replace the displayed result.</li>
          <li>Use <b>Restore working point</b> to restore settings and their verified
          checkpoint association. Check the status before continuing.</li>
          <li>Open <b>Advanced</b> for <b>Compare with current</b>, Pin A/Pin B
          and <b>Compare A / B</b>, or record import/export.</li>
        </ol>
        <h3>Choose what to save</h3>
        <ul>
          <li><b>Save operating profile...</b> preserves reusable instrument settings.</li>
          <li><b>Export result</b> / <b>Open result</b> use .temresult for a complete
          saved calculation and supported section continuation.</li>
          <li>Working-point Import/Export uses .temwp for the working-point record.</li>
          <li>Virtual-electron sessions use .temdiag and remain separate from
          physical result archives.</li>
        </ul>
        <h3>Read the result</h3>
        <p>Loading settings does not establish that an old result matches them.
        Inspect compatibility and captured inputs before reuse. Old complete results
        can remain visible while current inputs change. Numerical evidence is tied
        to its recorded scope, rather than to a filename or a successful load.</p>
        <p>See <a href="help:validation">validation and convergence</a>.</p>
        """,
    ),
    HelpTopic(
        "layout",
        "Physical Layout and workspace views",
        "Inspect component geometry, vacuum regions and the arrangement of your workspace.",
        ("geometry", "3D", "STEP", "CAD", "mechanical", "dimensions", "materials",
         "vacuum", "workspace", "layout", "dock", "reset"),
        """
        <p><b>Where:</b> <b>Physical Layout</b>, <b>Vacuum map</b>, and
        <b>View → Layouts</b>.</p>
        <ol>
          <li>In Physical Layout, use <b>2D</b>, <b>Rotating section</b>,
          <b>3D Parts</b> and <b>3D</b> to inspect the selected instrument.</li>
          <li>Select a component to inspect its dimensions, materials and available
          editing controls. Check whether an edit changes optical position,
          field geometry, an intercepting body or only presentation.</li>
          <li>Use <b>Vacuum map</b> to inspect the declared vacuum regions and
          their configured pressures.</li>
          <li>Arrange panels and splitters, then use <b>View → Layouts →
          Save current layout</b> or <b>Save layout as...</b>.
          <b>View → Reset workspace layout</b> restores the workspace arrangement.</li>
        </ol>
        <h3>Read the result</h3>
        <p>A workspace layout stores presentation choices; it does not restore an
        optical working point. Geometry edits can affect the physical calculation,
        so check result status afterwards.</p>
        <p>Mechanical drawings and imported shapes can contain illustrative or
        incompletely known internals. Their appearance alone does not establish a
        calibrated field. Vacuum pressures are supplied settings, not predictions
        of pumping performance.</p>
        <p>Use <b>File → Dimension definitions and evidence audit…</b> when checking
        dimension meanings and evidence. See <a href="help:instrument">component
        controls</a> and <a href="help:validation">model validation</a>.</p>
        """,
    ),
    HelpTopic(
        "validation",
        "Optical Transfer and model validation",
        "Distinguish configured models, measured simulator responses and numerical convergence.",
        ("validation", "convergence", "sampling", "mesh", "boundary", "field", "residual",
         "aberration", "Optical Transfer", "Model Inspector", "evidence", "accuracy"),
        """
        <p><b>Where:</b> <b>Optical Transfer</b>, <b>Model Inspector</b>,
        aberration views, and <b>Working points → Advanced → Sampling and Convergence</b>.</p>
        <ol>
          <li>Use Optical Transfer to inspect first-order transport, numeric details
          and mode comparisons for the selected optical configuration.</li>
          <li>In Model Inspector, select the component and inspect <b>Model checks</b>,
          <b>Magnetic circuits</b> and <b>Field validation</b>.</li>
          <li>Use <b>Check mesh and boundary</b> for the available magnetic-field
          convergence study; inspect its scope and results before exporting validation.</li>
          <li>In an aberration view, distinguish <b>Configured / inferred model</b>,
          <b>Actual incident beam</b> and <b>System residual</b>.
          Use <b>Measure system residual</b> for the current optical-model diagnostic.</li>
        </ol>
        <h3>Read the result</h3>
        <p>Configured coefficients describe selected model inputs. A system-residual
        measurement describes a finite diagnostic pupil and reference plane.
        Actual incident-beam statistics describe the executed particle population.
        These quantities answer different questions and need not agree outside their scope.</p>
        <p>Increase particle sampling and reduce integration steps independently
        when assessing numerical stability. Wave calculations also require appropriate
        spatial, energy and field sampling. A completed calculation, cache match,
        small fitted residual or passed local check does not establish whole-instrument
        accuracy. Keep unresolved and unavailable results visible in your conclusions.</p>
        <p><a href="help:correctors">Corrector channels</a> use local measured model
        responses; <a href="help:coherent_beam">phase observations</a> have additional
        sampling limits.</p>
        """,
    ),
    HelpTopic(
        "design",
        "Design Explorer and comparisons",
        "Compare captured settings and explore parameter sensitivity with explicit sweeps.",
        ("design", "experiment", "Capture A", "Capture B", "sweep", "sensitivity",
         "tolerance", "geometry candidates", "comparison"),
        """
        <p><b>Where:</b> <b>Design Explorer</b>, with <b>Design comparisons</b>
        and <b>CM / objective / scan</b> study areas.</p>
        <ol>
          <li>Set the instrument to a useful reference and click <b>Capture A</b>.</li>
          <li>Change the intended settings, then click <b>Capture B</b>.
          Inspect the captured-input and result-status differences.</li>
          <li>For a sweep, use <b>Add parameter</b>, enter a bounded range and
          choose the available measurement or tolerance settings.</li>
          <li>Click <b>Run sweep</b>. Inspect <b>Points / sensitivity</b> and
          <b>Plots / saved experiments</b>; use <b>Cancel</b> when needed.</li>
        </ol>
        <h3>Read the result</h3>
        <p>A capture records a set of inputs. It does not by itself calculate or
        validate the corresponding beam. Result and cache statuses tell you which
        evidence is available for each captured state.</p>
        <p>Sweeps show how the selected model and metric respond within the sampled
        range. Review failed or unresolved points and compare numerical settings
        before interpreting sensitivity. A favourable simulated trend is a candidate
        for further study, not evidence that a real microscope has been tuned.</p>
        <p>For durable records and comparison of complete calculations, use
        <a href="help:working_points">Working points and saved results</a>.</p>
        """,
    ),
)
