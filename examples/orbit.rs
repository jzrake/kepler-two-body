use kepler_two_body;

fn main()
{
	let elements    = kepler_two_body::OrbitalElements(1.0, 1.0, 1.0, 0.6);
	let orientation = kepler_two_body::OrbitalOrientation(0.0, 0.0, 0.0, 0.0, 0.1, 0.0);

	for i in 0..100 {
		let t = (i as f64) * 0.01 * 2.0 * std::f64::consts::PI;
		let state = elements.orbital_state_from_time_and_orientation(t, orientation);

		println!("{:+.8} {:+.8} {:+.8} {:+.8}",
			state.0.position_x(),
			state.0.position_y(),
			state.1.position_x(),
			state.1.position_y());
	}
}
