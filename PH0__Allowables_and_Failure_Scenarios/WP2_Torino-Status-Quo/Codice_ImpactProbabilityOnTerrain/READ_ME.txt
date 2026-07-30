================================================================================
PHASE 1 — TERRAIN MAP GENERATION
================================================================================

PURPOSE
-------
Phase 1 transforms a raw OpenStreetMap export into a georeferenced grid where
every cell is labelled with a terrain type. This grid is the geographic
foundation on which all subsequent+ phases operate. The output is a 2D integer
array — one value per cell — where each value encodes the dominant terrain type
at that location: GROUND, GREEN/NATURAL, ROAD, WATER, or BUILDING.


--------------------------------------------------------------------------------
STEP 1 — EXTRACTING AND CLASSIFYING GEOGRAPHIC FEATURES FROM THE OSM FILE
         (Block 1)
--------------------------------------------------------------------------------

The input is a .osm file, which is a plain XML document containing three types
of elements:

  - Nodes      : individual points with a latitude and longitude
  - Ways       : ordered lists of node references, representing either lines
                 (roads, rivers) or closed polygons (buildings, parks)
  - Relations  : groups of ways that together form a larger feature (e.g. the
                 Po river, which is too long to be stored as a single way)

The code parses this XML directly using Python's built-in xml.etree.ElementTree
library, without any external GIS dependencies. Every node's coordinates are
loaded into a lookup dictionary keyed by node ID, so that when a way references
its nodes, their coordinates can be resolved instantly.

Each way is then examined for its tags — key-value pairs that describe what the
feature is (e.g. building=yes, highway=residential, waterway=stream). Based on
these tags, the way is classified into one of five terrain classes and converted
into a Shapely geometry object:

  WATER
    Ways tagged with any waterway key are converted to Shapely Polygon objects
    if closed, or LineString objects with a small buffer if open. The buffer is
    applied in degrees at this stage (before metric reprojection) and is tuned
    per river: the Dora Riparia receives a narrower buffer (0.0002 deg ~ 18 m)
    than other waterways (0.0001 deg ~ 9 m). The Po river is additionally
    extracted from type=waterway relations, since it is stored in OSM as many
    short way segments joined by a relation rather than as a single way — each
    segment receives a buffer of 0.0007 deg ~ 60 m reflecting the river's real
    width.

  BUILDINGS
    Ways tagged with any recognised building value (yes, apartments, residential,
    house, garage, school, church, industrial, retail, commercial, office,
    university, hospital, and others) with closed geometry become Polygon objects
    representing footprints. The full list of recognised values was derived
    directly from the tag explorer output for the specific OSM file in use.

  ROADS
    Ways tagged with any recognised highway value (footway, residential, service,
    tertiary, secondary, cycleway, pedestrian, primary, steps, path, unclassified,
    busway, track, corridor, living_street and link variants) become LineString
    objects. Active railway lines (tram, rail, subway) are also collected as road
    geometries since they represent infrastructure corridors with similar risk
    implications. No buffer is applied at this stage — road and railway width is
    added later in metres after reprojection to UTM.

  GREEN / NATURAL
    Three separate OSM tag keys encode vegetation and open land, all of which are
    collected into the same green geometry class:

    leisure  : garden, pitch, park, playground, dog_park, track, schoolyard,
               nature_reserve
    landuse  : grass, farmland, flowerbed, meadow, recreation_ground, forest,
               allotments, vineyard, orchard, greenfield
    natural  : wood, scrub, grassland

    All three are checked independently using elif — a way matching any one of
    them is classified as green. Ways with closed geometry of at least 4 points
    become Polygon objects.

  GROUND
    The default class — assigned to any cell whose centre point does not fall
    inside any of the four explicitly classified terrain layers. Ground therefore
    represents all features that OSM has not tagged with a recognised value, or
    that are tagged with keys outside the four classification sets. In an urban
    area this includes untagged courtyards, construction sites, and any features
    present in the real world but absent from the OSM dataset. This is a known
    limitation of the centre-point classification approach and is acknowledged
    as such in the analysis.

At the end of Block 1, all individual geometries within each class are merged
into a single unified Shapely geometry using unary_union. This dissolves
internal boundaries between adjacent features of the same type and prepares
each class for efficient spatial querying in Block 3.


--------------------------------------------------------------------------------
STEP 2 — CONVERTING TO A METRIC COORDINATE SYSTEM (Block 2)
--------------------------------------------------------------------------------

All coordinates extracted from the OSM file are in WGS84 — the geographic
coordinate system used by GPS and OpenStreetMap, where positions are expressed
as latitude and longitude in degrees. Degrees are angles, not distances: at
Turin's latitude (45 deg N), one degree of longitude equals approximately 78 km
while one degree of latitude equals approximately 111 km. This asymmetry makes
degree coordinates unsuitable for any calculation involving physical distances.

The code reprojects all geometries to UTM Zone 32N (EPSG:32632), a projected
metric coordinate system covering northern Italy. In UTM, positions are
expressed as:

  Easting  (x) : distance in metres from the zone's central meridian (9 deg E
                 for Zone 32), with a false offset of 500,000 m to avoid
                 negative values
  Northing (y) : distance in metres from the equator

For Turin, this yields coordinates of approximately x ~ 399,500 m and
y ~ 4,991,800 m — meaning the city is roughly 100 km west of the zone's
central meridian and approximately 4,992 km north of the equator.

The reprojection is performed using the pyproj library, which applies the full
Transverse Mercator projection mathematics automatically. After this step,
every coordinate in the pipeline is in metres, and Euclidean distances between
any two points are physically exact.

NOTE: This is a foundational architectural decision. The physics simulation in
Phase 3 adds impact distances in metres to trajectory coordinates also in
metres, with no approximation. The entire pipeline shares one coordinate system
from this point forward.

The bounding box of the OSM export is also converted to UTM at this stage, and
the physical size of each grid cell is computed:

  cell width  = (x_max - x_min) / N_COLS
  cell height = (y_max - y_min) / N_ROWS

At 200x200 resolution over this area, each cell measures approximately
10 m x 6.5 m.


--------------------------------------------------------------------------------
STEP 3 — BUILDING AND CLASSIFYING THE GRID (Block 3)
--------------------------------------------------------------------------------

The grid is defined by two 2D arrays grid_x and grid_y, each of shape
(N_ROWS, N_COLS), where grid_x[i,j] and grid_y[i,j] give the UTM metre
coordinates of the centre point of cell (i,j). These arrays are computed in
Block 2 and carry no terrain information — they are purely a geometric lookup
table.

CLASSIFICATION APPROACH — CENTRE POINT SAMPLING
  Classification works by testing each cell's centre point against the four
  terrain geometry layers. For each cell (i,j), a Shapely Point is created at
  coordinates (grid_x[i,j], grid_y[i,j]) and queried against each terrain
  layer in sequence. If the point falls inside a terrain polygon, the cell
  receives that terrain label.

  This is a deliberate simplification: rather than computing what fraction of
  each cell's area is covered by each terrain type (a majority-vote approach),
  only the centre point is tested. This approximation is justified because at
  10 m cell resolution most cells fall cleanly within a single terrain type,
  and boundary errors affect only cells whose centre lands near a terrain edge.
  For a probabilistic risk map this level of precision is entirely acceptable.

PRIORITY ORDER
  Classification uses a strict priority order:

    green/natural -> road -> water -> building

  implemented by running all four checks with if rather than elif, so each
  label can overwrite the previous one. Building has the highest priority and
  always wins. Ground is the default — all cells start as ground (0) and are
  only overwritten if a match is found.

ROAD BUFFERING
  Since OSM stores roads as centrelines with no physical width, the entire road
  geometry union is expanded by a 5 metre buffer in UTM metres before
  classification. This gives roads a realistic 10 m total width on the grid.
  This buffer is applied in metres rather than degrees, making it exact.

SPATIAL INDEXING — STRtree
  Testing 40,000 cell centres against 1,400+ individual geometries naively
  would require millions of geometric intersection checks. To avoid this, each
  terrain layer is loaded into a Shapely STRtree (Sort-Tile-Recursive tree) — a
  spatial index that organises geometries into a hierarchy of bounding boxes.

  When querying a point, the tree eliminates entire branches of geometries that
  are geographically far from the query point in a few comparisons, reducing
  the number of precise intersection checks from hundreds to typically fewer
  than ten per cell. The tree is built once per terrain class and queried
  40,000 times, making Block 3 complete in tens of seconds rather than hours.

OUTPUT — terrain_grid
  A (N_ROWS x N_COLS) integer array encoding the terrain label of every cell:

    0 = Ground
    1 = Green/Natural
    2 = Road
    3 = Water
    4 = Building


--------------------------------------------------------------------------------
STEP 4 — VISUALISATION (Block 4)
--------------------------------------------------------------------------------

The terrain grid is rendered as a coloured image using Matplotlib, with UTM
metre coordinates used as axis extents so the map is correctly georeferenced.
Each integer label is mapped to a distinct colour. The result is saved as
terrain_map.png.

All outputs needed by subsequent phases are serialised to terrain_stage1.pkl:

  - terrain_grid   : (N_ROWS x N_COLS) integer array of terrain labels
  - grid_x, grid_y : (N_ROWS x N_COLS) arrays of cell centre UTM coordinates
  - x_min, x_max   : easting bounds in metres
  - y_min, y_max   : northing bounds in metres
  - cell_w, cell_h : physical cell dimensions in metres
  - N_ROWS, N_COLS : grid resolution
  - LAT/LON bounds : original bounding box in degrees


--------------------------------------------------------------------------------
KEY DESIGN DECISIONS
--------------------------------------------------------------------------------

  No external GIS library
    The OSM file is parsed directly with xml.etree.ElementTree, avoiding
    dependency on geopandas or fiona.

  Single coordinate system from Block 2 onward
    All geometry, grid coordinates, trajectory points, and physics operate
    in UTM Zone 32N metres throughout the entire pipeline.

  Resolution independence
    The grid resolution N_ROWS / N_COLS is the only parameter that needs
    changing to adjust detail vs speed. All other quantities derive from it
    automatically.

  Centre-point classification
    A known simplification, justified by cell size and the probabilistic
    nature of the downstream analysis.




================================================================================
PHASE 2 — TRAJECTORY DEFINITION, SPLINE INTERPOLATION AND VISUALISATION
================================================================================

PURPOSE
-------
Phase 2 defines the drone's flight trajectory over the terrain map produced in
Phase 1. The trajectory is specified by a small set of geographic waypoints
defined by the user, interpolated into a smooth continuous curve using a
parametric spline, and sampled at fixed physical spacing to produce the sequence
of points that Phase 3 will use as drop locations for the Monte Carlo
simulation. A key design requirement is that the trajectory is entirely
resolution-independent: it is defined once in real-world metre coordinates and
remains identical regardless of the grid resolution chosen in Phase 1 or the
number of Monte Carlo samples used in Phase 3.


--------------------------------------------------------------------------------
STEP 1 — LOADING PHASE 1 OUTPUTS AND WAYPOINT INPUT (Block 1)
--------------------------------------------------------------------------------

Phase 2 begins by loading the terrain grid and all geographic metadata saved
by Phase 1 into terrain_stage1.pkl. The bounding box (in both degrees and UTM
metres) is needed for waypoint validation. The terrain grid is needed for
the final visualisation.

WAYPOINT INPUT FILE
  The trajectory is defined externally in a plain Excel file (waypoints.xlsx)
  with three columns:

    lat    : latitude in decimal degrees (WGS84)
    lon    : longitude in decimal degrees (WGS84)
    name   : optional label for identification and map annotation

  This is a deliberate design choice. The trajectory is a fixed real-world
  input that never changes regardless of grid resolution or simulation
  parameters. Storing it in a separate file means the trajectory can be edited
  without touching any code.

SAMPLE SPACING
  A single parameter SAMPLE_SPACING_M (default: 10.0 m) controls how densely
  the spline will be sampled for the Monte Carlo loop in Phase 3 — one
  simulation point every 10 metres along the trajectory. This parameter is
  completely independent of the grid resolution.

WAYPOINT VALIDATION
  Every waypoint is checked against the bounding box of the OSM export. If a
  waypoint falls outside the terrain map, the spline would pass through an area
  with no terrain classification, making Phase 3 results meaningless for those
  segments. Out-of-bounds waypoints are dropped with a printed warning rather
  than raising a hard error, so a partially valid set still produces a usable
  trajectory. If fewer than 2 valid waypoints remain after filtering, the code
  stops with an explicit error message.

COORDINATE CONVERSION
  Waypoints are defined in lat/lon degrees — easy to read off any map tool such
  as Google Maps — and immediately converted to UTM Zone 32N (EPSG:32632) in
  metres using the same pyproj transformer as Phase 1. From this point forward,
  the original lat/lon values are no longer used. All trajectory geometry
  operates in UTM metres, consistent with the terrain grid and the physics
  simulation in Phase 3.


--------------------------------------------------------------------------------
STEP 2 — SPLINE FITTING AND UNIFORM SAMPLING (Block 2)
--------------------------------------------------------------------------------

PARAMETRIC CUBIC B-SPLINE
  The waypoints are interpolated using a parametric B-spline fitted by SciPy's
  splprep function. Two properties of this approach are essential:

  Parametric
    A standard spline fits y as a function of x, which breaks down if the curve
    doubles back on itself — as a river-following trajectory typically does.
    A parametric spline instead expresses both x and y as functions of an
    independent parameter t (ranging from 0 to 1 along the curve):

      x = f(t)
      y = g(t)

    This allows the curve to take any shape — including U-turns, loops and
    sharp bends — without any mathematical problems.

  Cubic (k=3)
    Degree k=3 means the curve is made of cubic polynomial pieces joined
    smoothly at each waypoint, with continuous first and second derivatives.
    This produces a smooth, natural-looking curve with no kinks or sudden
    changes in curvature. The code falls back to k=1 (straight line segments
    between waypoints) if fewer than 4 valid waypoints are available, since
    a cubic spline requires at least 4 control points.

  B-spline with s=0
    B-spline (basis spline) uses a set of locally-supported basis functions,
    each influencing only a small region of the curve, rather than one global
    polynomial. This avoids the numerical instability that arises from fitting
    a single high-degree polynomial through many points (Runge's phenomenon).
    With s=0, the spline is forced to pass exactly through every waypoint —
    no smoothing is applied.

UNIFORM ARC-LENGTH SAMPLING
  The spline parameter t is not inherently proportional to physical distance —
  equal steps in t do not correspond to equal distances along the curve. On a
  tight bend, equal steps in t produce smaller physical steps; on a straight
  section, equal steps produce larger ones. Arc-length reparameterisation is
  therefore required to achieve uniform physical spacing.

  The process works in four stages:

  STAGE 1 — Dense evaluation
    The spline is evaluated at 100,000 equally spaced t values between 0 and 1:

      t = 0.00000, 0.00001, 0.00002, ..., 1.00000

    For each t value, splev returns a physical coordinate pair (x, y) in UTM
    metres. This produces a dense polyline of 100,000 points approximating the
    continuous curve. At a total trajectory length of ~11 km, consecutive points
    are approximately 0.11 m apart — fine enough that the polyline is
    indistinguishable from the true curve at the 10 m sampling scale.

  STAGE 2 — Cumulative arc length computation
    The physical distance between each consecutive pair of points is computed
    using Pythagoras:

      segment_length(i) = sqrt( (x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2 )

    Cumulative summation then builds a mapping from t value to distance along
    the curve:

      t_0     →   0.00 m
      t_1     →   0.11 m
      t_2     →   0.22 m
      ...
      t_99999 →  10989.51 m   (total arc length)

    This cumulative distance vector is the key lookup table for the next stage.
    Note that physical distance is NOT computed analytically from the polynomial
    expression — it is approximated numerically by summing the lengths of the
    100,000 tiny straight segments. At 0.11 m resolution the approximation error
    is negligible relative to the 10 m sampling spacing.

  STAGE 3 — Interpolation to find t values at uniform distances
    The target distances are defined as:

      d = 0, 10, 20, 30, ..., total_length   metres

    For each target distance d, the corresponding t value is found by linear
    interpolation on the cumulative distance lookup table. Concretely, given a
    target distance d = 10 m, the two bracketing entries in the table are
    identified:

      point A:  cumulative distance = 8.73 m  →  t_A = 0.00030
      point B:  cumulative distance = 11.60 m →  t_B = 0.00040

    The interpolation weight is:

      b = (d - d_A) / (d_B - d_A)
        = (10 - 8.73) / (11.60 - 8.73)
        = 0.443

    The corresponding t value is:

      t* = t_A + b x (t_B - t_A)
         = 0.00030 + 0.443 x 0.00010
         = 0.000344

    This is performed in a single vectorised operation using numpy's np.interp
    function — no loop is required. The result u_uniform is the complete array
    of t values corresponding to distances 0, 10, 20, 30... metres along the
    curve.

    An important subtlety: np.interp interpolates in t space, not in coordinate
    space. The interpolated quantity is the t value, not the (x, y) coordinates
    directly. This distinction matters on curved sections — linearly
    interpolating coordinates would cut across the curve, while interpolating t
    and then evaluating the spline follows the curve exactly.

  STAGE 4 — Final evaluation at uniform t values
    The spline is evaluated at u_uniform using splev, returning the final
    sequence of (traj_x, traj_y) coordinate pairs in UTM metres. Consecutive
    points are separated by exactly SAMPLE_SPACING_M metres measured along the
    curve — not in a straight line between them.

    Resolution ratio check: with 100,000 dense t values over ~11 km, the
    average spacing between consecutive dense points is ~0.11 m. The requested
    sample spacing is 10 m. The resolution ratio is therefore:

      10 m / 0.11 m = 91x

    This means approximately 91 dense table entries sit between each 10 m
    sample point, making the interpolation highly accurate. A ratio above 10x
    is considered safe; 91x provides a large safety margin.

  The result is a sequence of (traj_x, traj_y) coordinate pairs in UTM metres,
  spaced exactly SAMPLE_SPACING_M metres apart along the curve. This sequence
  is fully resolution-independent: changing the terrain grid resolution or the
  Monte Carlo sample count does not affect these coordinates in any way.

HEADING ANGLE COMPUTATION
  At each sample point, the drone's heading angle (direction of flight) is
  computed from the local tangent to the spline:

    dx = gradient(traj_x)
    dy = gradient(traj_y)
    theta = arctan2(dy, dx)   [degrees, measured from East]

  This heading array is passed to Phase 3 so that the Monte Carlo impact cloud
  at each trajectory point is correctly oriented along the flight direction
  rather than always pointing in a fixed direction. On a curved trajectory this
  makes a meaningful difference to which side of the river receives higher
  impact probability.


--------------------------------------------------------------------------------
STEP 3 — VISUALISATION (Block 3)
--------------------------------------------------------------------------------

The trajectory is overlaid on the Phase 1 terrain map using Matplotlib. The
visualisation shows:

  - The terrain base map (same colours as Phase 1)
  - The full interpolated trajectory as a solid black line
  - The original waypoints as yellow dots with name labels

The UTM metre extents are used as axis bounds so the trajectory is correctly
georeferenced on top of the terrain grid. The result is saved as
terrain_with_trajectory.png.


--------------------------------------------------------------------------------
STEP 4 — SAVING OUTPUTS (Block 4)
--------------------------------------------------------------------------------

All trajectory data needed by Phase 3 is saved to trajectory_stage2.pkl:

  - traj_x, traj_y     : UTM metre coordinates of all sampled points
  - traj_theta_deg     : heading angle in degrees at each sample point
  - wp_x, wp_y         : UTM metre coordinates of the original waypoints
  - wp_names           : waypoint name labels
  - sample_spacing     : physical spacing between sample points (metres)
  - total_length_m     : total arc length of the trajectory (metres)
  - n_samples          : total number of sampled trajectory points


--------------------------------------------------------------------------------
KEY DESIGN DECISIONS
--------------------------------------------------------------------------------

  Resolution-independent trajectory
    The spline is defined in UTM metres and sampled at fixed physical spacing.
    Changing the terrain grid resolution in Phase 1 or the Monte Carlo sample
    count in Phase 3 has no effect on the trajectory coordinates.

  External waypoint file
    The trajectory is stored in waypoints.xlsx, separate from the code.
    This makes it easy to modify the flight path without touching the pipeline.

  Cubic spline with k=1 fallback
    Cubic interpolation (k=3) requires at least 4 waypoints. If fewer valid
    waypoints are available after bounding box filtering, the code automatically
    falls back to linear interpolation (k=1) to avoid a hard failure.

  Heading from spline tangent
    The drone heading at each trajectory point is derived analytically from the
    spline tangent rather than hardcoded, ensuring the Monte Carlo impact
    distribution is correctly oriented along the actual flight direction at
    every point.


================================================================================
PHASE 3 — MONTE CARLO BALLISTIC IMPACT PROBABILITY MAP
================================================================================

PURPOSE
-------
Phase 3 computes the probability that the medical payload, if separated from
the drone at any point along the trajectory defined in Phase 2, will impact
each geographic cell of the map defined in Phase 1. The output is a
normalised 2D probability map — a heatmap — where each cell contains the
fraction of all possible impact scenarios that land there, accumulated across
the entire flight path.


--------------------------------------------------------------------------------
STEP 1 — SIMULATION PARAMETERS (Block 2)
--------------------------------------------------------------------------------

OUTPUT GRID
  The impact probability heatmap uses its own independent grid resolution,
  defined by N1 (cells in easting) and N2 (cells in northing). This is
  completely independent of the terrain grid resolution defined in Phase 1.
  The geographic extent is identical — the same UTM metre bounding box — but
  the cell size can differ. Higher resolution produces finer detail in the
  probability map at the cost of memory.

PHYSICAL PARAMETERS (deterministic — fixed across all simulations)
  These describe the invariant physical properties of the payload and
  environment:

    g          = 9.81   m/s²   gravitational acceleration
    densita    = 1.225  kg/m³  air density at sea level
    massa      = 1.5    kg     payload mass
    superficie = 0.04   m²     frontal reference area of the payload

  The drag constant c is derived from these:

    c = 0.5 × superficie × densita × Cd

  where Cd is the drag coefficient (uncertain — sampled randomly, see below).
  This constant appears directly in the equations of motion and controls how
  strongly aerodynamic drag decelerates the payload during descent.

UNCERTAIN PARAMETERS (stochastic — sampled randomly each simulation)
  Real-world conditions at the moment of payload separation are not known
  exactly. Each uncertain parameter is modelled as a Gaussian distribution
  described by a mean (mu) and standard deviation (sigma):

    altezza_mu / altezza_sigma   : altitude at separation (m)
                                   reflects small variations in flight altitude
    Cd_mu / Cd_sigma             : aerodynamic drag coefficient
                                   reflects payload tumbling or rotation
    Vx_i_mu / Vx_i_sigma        : horizontal velocity at separation (m/s)
                                   inherited from drone cruise speed
    Vy_i_mu / Vy_i_sigma        : vertical velocity at separation (m/s)
                                   mean zero assumes level flight at failure
    mu_vento / sigma_vento       : wind speed (m/s)
    mu_drone / sigma_drone       : wind direction (degrees)
                                   large sigma produces wide lateral spread

MONTE CARLO SAMPLE COUNT
  GUESS defines the number of independent simulations run at each trajectory
  point. Each simulation draws a fresh random sample of all uncertain
  parameters above and computes one complete ballistic trajectory. Increase
  GUESS to 1000 or more for statistically meaningful results. Values of 30
  or below are suitable only for functional testing.


--------------------------------------------------------------------------------
STEP 2 — THE MONTE CARLO METHOD (conceptual background)
--------------------------------------------------------------------------------

The impact location of the payload cannot be predicted as a single point
because the conditions at the moment of failure are uncertain. Instead of
computing one trajectory with fixed inputs, the Monte Carlo method computes
a large number of trajectories — each one using a different random
realisation of the uncertain inputs drawn from their probability
distributions.

For each trajectory point along the drone's flight path:

  1. Draw GUESS random samples of altitude, drag coefficient, initial
     velocities, wind speed and wind direction simultaneously
  2. Compute one complete ballistic descent for each sample
  3. Record where each descent lands on the ground
  4. Count how many of the GUESS impacts fall in each grid cell
  5. Divide by GUESS to get the probability of impact per cell

The result is a probability density function (PDF) over the ground — not a
single impact point but a cloud of possible impact locations, each with an
associated probability. The shape of this cloud depends on the spread of the
uncertain inputs: wide distributions produce diffuse clouds, narrow
distributions produce tight ones.

This process is repeated independently at every trajectory point. The
individual PDFs are then accumulated into a single global map by simple
addition, since all trajectory points are assumed equally likely to be the
point of failure. The accumulated map is normalised at the end so that the
sum of all cell probabilities equals exactly 1.0.


--------------------------------------------------------------------------------
STEP 3 — THE PHYSICS MODEL (ballistic descent with aerodynamic drag)
--------------------------------------------------------------------------------

THE DRAG FORCE
  The aerodynamic drag force acting on the payload at any instant is:

    F_drag = c × V²

  where c = 0.5 × S × rho × Cd and V is the instantaneous speed. Drag
  always opposes motion — it acts against horizontal velocity, against upward
  vertical velocity, and against downward vertical velocity. Because drag
  depends on the square of velocity, the equations of motion are nonlinear
  and cannot be solved with simple projectile formulas.

  The model solves these equations ANALYTICALLY — deriving exact closed-form
  mathematical expressions for position and velocity at any time. This is why
  the helper functions contain expressions involving log, arctan, tanh and
  arctanh — these are the exact solutions to the drag-modified equations of
  motion, not numerical approximations.

TERMINAL VELOCITY
  The most important derived quantity is the terminal velocity:

    Gamma = sqrt(m × g / c)

  This is the maximum downward speed the payload can reach, where drag
  exactly balances gravity. It controls everything about how drag affects
  the trajectory:

    High Gamma (heavy, small, low Cd) : drag is weak, payload lands far away
    Low Gamma  (light, large, high Cd): drag is strong, payload lands nearby

  For the payload in this model (m=1.5 kg, S=0.04 m², Cd~1.0):

    c     = 0.5 × 0.04 × 1.225 × 1.0 ≈ 0.0245
    Gamma = sqrt(1.5 × 9.81 / 0.0245) ≈ 24.5 m/s  (~88 km/h)

  Since the drone cruise speed (~23 m/s) is approximately equal to terminal
  velocity, horizontal and vertical velocities are of similar magnitude at
  impact, producing impact angles of roughly 45 degrees from horizontal.

THE THREE DESCENT PHASES
  The analytical solution is split into three sequential phases, each with
  its own governing equations:

  Phase 1 — Upward motion (if payload has upward velocity at separation)
    Both gravity and drag act downward, opposing the upward motion.
    The payload decelerates faster than in free flight, reaching peak
    altitude sooner and lower than a simple projectile would. Horizontal
    velocity decays continuously according to:

      Vx(t) = m × Vx_i / (m + Vx_i × c × t)

    Horizontal distance covered during this phase: x1
    Duration of this phase: t_top_hat
    If Vy_i <= 0 at separation, this phase is skipped entirely.

  Phase 2 — Descent from peak to Vx = Vy crossing
    Gravity accelerates the payload downward while drag increasingly
    resists this, limiting vertical speed toward terminal velocity Gamma.
    Meanwhile horizontal velocity continues to decay. This phase ends when
    horizontal and vertical velocity magnitudes become equal — a natural
    regime boundary in the analytical solution.

    Horizontal distance covered during this phase: x2
    Duration: tc_hat - t_top_hat

  Phase 3 — Final descent from crossing to ground impact
    After the Vx = Vy crossing, both velocity components are coupled through
    the drag force and a different set of analytical expressions applies.
    The payload continues losing horizontal speed while vertical speed
    approaches terminal velocity.

    Horizontal distance covered during this phase: x3
    Duration: t_im - tc_hat

  Total horizontal range from separation point to impact:

    x = x1 + x2 + x3

  Total time of flight:

    t_im = t_top_hat + t_drop_hat

HOW DRAG AFFECTS THE IMPACT LOCATION
  Without drag the horizontal range would simply be Vx_i × t_fall. With
  drag, the actual range is always shorter because:

    1. Horizontal velocity decays continuously throughout all three phases
    2. Drag slows vertical acceleration, increasing flight time, which gives
       more time for horizontal deceleration to act
    3. Wind drift becomes relatively more important as horizontal speed
       decreases — the payload drifts laterally more than a fast projectile

  This produces an asymmetric impact distribution that is sensitive to the
  ratio between initial horizontal speed, terminal velocity, and wind speed.

VECTORISATION
  All computations are fully vectorised over the GUESS dimension — every
  variable is a NumPy array of length GUESS and all GUESS simulations run
  simultaneously as array operations rather than in a Python loop. This
  makes the function orders of magnitude faster than a naive per-simulation
  loop.

  One exception: the drag coefficient Cd is drawn as a single scalar shared
  across all GUESS simulations within one call. This matches the original
  MATLAB behaviour and reflects the assumption that Cd is a property of the
  payload shape — uncertain between events but fixed within one failure event.


--------------------------------------------------------------------------------
STEP 4 — THE MAIN LOOP AND PDF ACCUMULATION (Block 3)
--------------------------------------------------------------------------------

For each of the n_samples trajectory points:

  1. Read the UTM metre coordinates (cx, cy) and heading angle theta_i
     from the Phase 2 outputs
  2. Call impact_point_ballistic_descent with (cx, cy) as the drop centre
     and theta_i as the drone heading
  3. Inside the function:
       a. Sample all uncertain inputs randomly (GUESS realisations)
       b. Run compute_impact_point_ballistic_descent — the vectorised
          physics model — to get horizontal range x and impact time t_im
          for all GUESS simulations simultaneously
       c. Rotate the impact cloud by theta_i so it is oriented along the
          actual flight direction rather than always pointing East
       d. Add wind drift: each impact point is shifted by
          w × t_im × [cos(phi), sin(phi)] where w is wind speed and phi
          is wind direction
       e. Translate to real-world UTM coordinates by adding (cx, cy)
       f. Bin all GUESS impact points into the N1 × N2 grid using a 2D
          histogram to produce the local PDF
  4. Add the local PDF to global_pdf (simple accumulation — equal weight
     at every trajectory point)
  5. Collect raw Vx, Vy and impact angle arrays for distribution plots

After the loop, global_pdf is normalised:

    global_pdf = global_pdf / global_pdf.sum()

Each cell value then represents the fraction of all possible impact
scenarios across the entire trajectory that land in that cell. Multiplying
by 100 gives the percentage probability.


--------------------------------------------------------------------------------
STEP 5 — OUTPUTS (Blocks 4 and 5)
--------------------------------------------------------------------------------

HEATMAP (impact_heatmap.png)
  The normalised probability map is overlaid on the Phase 1 terrain map
  using a hot_r colormap (dark = high probability, light = low). The drone
  trajectory and waypoints are drawn on top for geographic reference.
  Colorbar units are percentage probability (%).

IMPACT CONDITION DISTRIBUTIONS (chart_impact_velocity_pdf.png)
  Three probability density function plots showing the distribution of
  impact conditions accumulated across all simulations and all trajectory
  points. The total sample count per plot is n_samples x GUESS. The y-axis
  of each plot shows probability density — not probability directly. Since
  probability density has units of 1/x-axis-unit, values can exceed 1 when
  the distribution is narrow. The total area under each curve always
  integrates to exactly 1.0.

    Vx at impact  : horizontal velocity at ground impact (m/s).
                    Reflects the combined effect of initial drone cruise
                    speed and aerodynamic drag during descent. Values are
                    always positive (forward motion).

    Vy at impact  : vertical velocity at ground impact (m/s).
                    Negative values indicate downward motion. The magnitude
                    is bounded by the terminal velocity Gamma = sqrt(mg/c),
                    which for this payload is approximately 24.5 m/s.
                    The distribution is concentrated near terminal velocity
                    for high-altitude separation events.

    Impact time   : total time of flight from payload separation to ground
                    impact (s). Includes both the upward phase (if any) and
                    the full descent phase. The distribution reflects the
                    combined uncertainty in separation altitude, drag
                    coefficient, and initial vertical velocity. A narrow
                    distribution indicates that impact time is primarily
                    driven by altitude rather than by the other uncertain
                    parameters.

TERRAIN IMPACT HISTOGRAM (chart_impact_by_terrain.png)
  A bar chart showing the cumulative impact probability attributed to each
  of the five terrain classes. For each cell in the impact grid, the
  terrain label is looked up from the Phase 1 terrain grid (the two grids
  share the same geographic bounding box but may have different resolutions,
  so each impact cell is mapped to the nearest terrain cell). The
  probability mass of all impact cells belonging to each terrain class is
  then summed and expressed as a percentage. This chart directly answers
  the question of which terrain type absorbs the highest share of impact
  risk along the simulated trajectory.

SAVED DATA (impact_stage3.pkl)
  The global probability map and grid metadata are saved for any further
  post-processing:

    global_pdf   : (N1 x N2) normalised probability array
    N1, N2       : output grid dimensions
    x_limits     : easting bounds in metres
    y_limits     : northing bounds in metres


--------------------------------------------------------------------------------
KEY DESIGN DECISIONS
--------------------------------------------------------------------------------

  Equal failure probability along trajectory
    Every trajectory point is weighted equally when accumulating PDFs.
    This assumes the drone has a uniform probability of failure at any
    point along the flight path. Non-uniform failure probability (e.g.
    higher risk over buildings) could be introduced by weighting each
    PDF before accumulation.

  Analytical physics model
    The ballistic descent is solved analytically rather than numerically.
    This is faster, exact, and avoids numerical integration errors. The
    cost is a more complex set of equations split across three phases.

  Per-point heading angle
    The drone heading theta is taken from the spline tangent at each
    trajectory point rather than being hardcoded. This correctly orients
    the impact cloud along the actual flight direction at every point,
    which matters significantly on curved trajectories.

  Resolution-independent output grid
    The impact heatmap grid (N1 × N2) is independent of the terrain grid
    from Phase 1. Both grids share the same geographic bounding box but
    can have different resolutions.

================================================================================