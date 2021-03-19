use std::f64::consts::PI;
static G: f64 = 1.0;




// ============================================================================
#[derive(Debug)]
pub struct UnboundOrbitalState(OrbitalState);

impl std::fmt::Display for UnboundOrbitalState {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "unbound orbital state, energy = {:+.6e}", self.0.total_energy())
    }
}

impl std::error::Error for UnboundOrbitalState {}




/**
 * Represents a single gravitating point mass: m, x, y, vx, vy
 */
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature="hdf5", repr(C), derive(hdf5::H5Type))]
#[cfg_attr(feature="serde", derive(serde::Serialize, serde::Deserialize))]

pub struct PointMass(pub f64, pub f64, pub f64, pub f64, pub f64);




/**
 * Represents two point masses on a bound orbit
 */
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature="hdf5", repr(C), derive(hdf5::H5Type))]
#[cfg_attr(feature="serde", derive(serde::Serialize, serde::Deserialize))]

pub struct OrbitalState(pub PointMass, pub PointMass);




/**
 * Represents the orbital elements of a two-body orbit: a, M, q, e
 */
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature="hdf5", repr(C), derive(hdf5::H5Type))]
#[cfg_attr(feature="serde", derive(serde::Serialize, serde::Deserialize))]

pub struct OrbitalElements(pub f64, pub f64, pub f64, pub f64);




/**
 * Represents the 2d orientation of a binary system, with a CM position and
 * velocity, an eccentricity vector (periapse argument), and a time of last
 * periapse
 */
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature="hdf5", repr(C), derive(hdf5::H5Type))]
#[cfg_attr(feature="serde", derive(serde::Serialize, serde::Deserialize))]

pub struct OrbitalOrientation(pub f64, pub f64, pub f64, pub f64, pub f64, pub f64);




/**
 * Represents an extended orbit as a combination of the orbital elements and the
 * orbital orientation
 */
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature="hdf5", repr(C), derive(hdf5::H5Type))]
#[cfg_attr(feature="serde", derive(serde::Serialize, serde::Deserialize))]

pub struct OrbitalParameters(pub OrbitalElements, pub OrbitalOrientation);




// ============================================================================
impl PointMass {
    pub fn mass          (self) -> f64 { self.0 }
    pub fn position_x    (self) -> f64 { self.1 }
    pub fn position_y    (self) -> f64 { self.2 }
    pub fn velocity_x    (self) -> f64 { self.3 }
    pub fn velocity_y    (self) -> f64 { self.4 }
}




// ============================================================================
impl OrbitalElements {
    pub fn semimajor_axis(self) -> f64 { self.0 }
    pub fn total_mass    (self) -> f64 { self.1 }
    pub fn mass_ratio    (self) -> f64 { self.2 }
    pub fn eccentricity  (self) -> f64 { self.3 }
}




// ============================================================================
impl OrbitalOrientation {
    pub fn cm_position_x    (self) -> f64 { self.0 }
    pub fn cm_position_y    (self) -> f64 { self.1 }
    pub fn cm_velocity_x    (self) -> f64 { self.2 }
    pub fn cm_velocity_y    (self) -> f64 { self.3 }
    pub fn periapse_argument(self) -> f64 { self.4 }
    pub fn periapse_time    (self) -> f64 { self.5 }
}




// ============================================================================
impl OrbitalParameters {
    pub fn elements    (self) -> OrbitalElements    { self.0 }
    pub fn orientation (self) -> OrbitalOrientation { self.1 }
}




// ============================================================================
fn solve_newton_rapheson<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(f: F, g: G, mut x: f64) -> f64 {
    while f64::abs(f(x)) > 1e-12 {
        x -= f(x) / g(x);
    }
    return x;
}

fn clamp_between_zero_and_one(x: f64) -> f64 {
    f64::min(1.0, f64::max(0.0, x))
}




// ============================================================================
impl PointMass {

    pub fn kinetic_energy(self) -> f64 {
        let vx = self.velocity_x();
        let vy = self.velocity_y();
        0.5 * self.mass() * (vx * vx + vy * vy)
    }

    pub fn angular_momentum(self) -> f64 {
        let x  = self.position_x();
        let y  = self.position_y();
        let vx = self.velocity_x();
        let vy = self.velocity_y();
        self.mass() * (x * vy - y * vx)
    }

    pub fn gravitational_potential(self, x: f64, y: f64, softening_length: f64) -> f64 {
        let dx = x - self.position_x();
        let dy = y - self.position_y();
        let r2 = dx * dx + dy * dy;
        let s2 = softening_length.powi(2);
        -G * self.mass() / (r2 + s2).sqrt()
    }

    pub fn gravitational_acceleration(self, x: f64, y: f64, softening_length: f64) -> [f64; 2] {
        let dx = x - self.position_x();
        let dy = y - self.position_y();
        let r2 = dx * dx + dy * dy;
        let s2 = softening_length.powi(2);
        let ax = -G * self.mass() / (r2 + s2).powf(1.5) * dx;
        let ay = -G * self.mass() / (r2 + s2).powf(1.5) * dy;
        [ax, ay]
    }

    pub fn perturb_mass(self, dm: f64) -> PointMass {
        let mut p = self;
        p.0 += dm;
        p
    }

    pub fn perturb_momentum(self, dpx: f64, dpy: f64) -> PointMass {
        let mut p = self;
        p.3 += dpx / self.mass();
        p.4 += dpy / self.mass();
        p
    }

    pub fn perturb_mass_and_momentum(self, dm: f64, dpx: f64, dpy: f64) -> PointMass {
        let mut p = self;
        p.0 += dm;
        p.3 += (dpx - self.velocity_x() * dm) / self.mass(); // dv = (dp - v dm) / m
        p.4 += (dpy - self.velocity_y() * dm) / self.mass();
        p
    }
}




// ============================================================================
impl OrbitalState {

    pub fn total_mass(self) -> f64 {
        self.0.mass() + self.1.mass()
    }

    pub fn mass_ratio(self) -> f64 {
        self.1.mass() / self.0.mass()
    }

    pub fn separation(self) -> f64 {
        let x1 = self.0.position_x();
        let y1 = self.0.position_y();
        let x2 = self.1.position_x();
        let y2 = self.1.position_y();
        f64::sqrt((x2 - x1).powi(2) + (y2 - y1).powi(2))
    }

    pub fn kinetic_energy(self) -> f64 {
        self.0.kinetic_energy() + self.1.kinetic_energy()
    }

    pub fn angular_momentum(self) -> f64 {
        self.0.angular_momentum() + self.1.angular_momentum()
    }

    pub fn gravitational_potential(self, x: f64, y: f64, softening_length: f64) -> f64 {
        self.0.gravitational_potential(x, y, softening_length) + self.1.gravitational_potential(x, y, softening_length)
    }

    pub fn total_energy(self) -> f64 {
        self.kinetic_energy() - G * self.0.mass() * self.1.mass() / self.separation()
    }

    /**
     * Transform this orbital state vector so that it has the new orientation.
     * This function rotates the position and velocity vectors according to the
     * argument of periapse, and translates them according to the center-of-mass
     * position and velocity. Note that the time-of-last-periapse-passage is
     * technically part of the orbital orientation, but is ignored by this
     * function, as that would change the intrinsic orbital phase.
     *
     * # Arguments
     * * o      - The orbital orientation to transform to
     */
    pub fn transform(self, o: OrbitalOrientation) -> OrbitalState {
        let m1  = self.0.mass();
        let x1  = self.0.position_x();
        let y1  = self.0.position_y();
        let vx1 = self.0.velocity_x();
        let vy1 = self.0.velocity_y();
        let m2  = self.1.mass();
        let x2  = self.1.position_x();
        let y2  = self.1.position_y();
        let vx2 = self.1.velocity_x();
        let vy2 = self.1.velocity_y();

        let c = f64::cos(-o.periapse_argument());
        let s = f64::sin(-o.periapse_argument());

        let x1p  =  x1  * c + y1  * s + o.cm_position_x();
        let y1p  = -x1  * s + y1  * c + o.cm_position_y();
        let x2p  =  x2  * c + y2  * s + o.cm_position_x();
        let y2p  = -x2  * s + y2  * c + o.cm_position_y();
        let vx1p =  vx1 * c + vy1 * s + o.cm_velocity_x();
        let vy1p = -vx1 * s + vy1 * c + o.cm_velocity_y();
        let vx2p =  vx2 * c + vy2 * s + o.cm_velocity_x();
        let vy2p = -vx2 * s + vy2 * c + o.cm_velocity_y();

        let c1 = PointMass(m1, x1p, y1p, vx1p, vy1p);
        let c2 = PointMass(m2, x2p, y2p, vx2p, vy2p);

        OrbitalState(c1, c2)
    }

    /**
     * Rotate an orbital state vector by the given angle: positive angle means
     * that the argument of periapse moves counter-clockwise, in other words
     * this function rotates the binary, not the coordinates.
     *
     * * angle  - The angle to rotate by
     *
     */
    pub fn rotate(self, angle: f64) -> OrbitalState {
        let orientation = OrbitalOrientation(0.0, 0.0, 0.0, 0.0, angle, 0.0);
        self.transform(orientation)
    }

    /**
     * Return a new orbital state vector if this one is perturbed by the given
     * masses and momenta.
     *
     * * dm1  -    Mass added to the primary
     * * dm2  -    Mass added to the secondary
     * * dpx1 -    Force (x) added to the primary
     * * dpx2 -    Force (x) added to the secondary
     * * dpy1 -    Force (y) added to the primary
     * * dpy2 -    Force (y) added to the secondary
     */
    pub fn perturb(self, dm1: f64, dm2: f64, dpx1: f64, dpx2: f64, dpy1: f64, dpy2: f64) -> Self {
        Self(self.0.perturb_mass_and_momentum(dm1, dpx1, dpy1), self.1.perturb_mass_and_momentum(dm2, dpx2, dpy2))
    }

    pub fn recover_orbital_parameters(self, t: f64) -> Result<OrbitalParameters, UnboundOrbitalState> {
        let c1 = self.0;
        let c2 = self.1;

        // component masses, total mass, and mass ratio
        let m1 = c1.mass();
        let m2 = c2.mass();
        let m = m1 + m2;
        let q = m2 / m1;

        // position and velocity of the CM frame
        let x_cm  = (c1.position_x() * c1.mass() + c2.position_x() * c2.mass()) / m;
        let y_cm  = (c1.position_y() * c1.mass() + c2.position_y() * c2.mass()) / m;
        let vx_cm = (c1.velocity_x() * c1.mass() + c2.velocity_x() * c2.mass()) / m;
        let vy_cm = (c1.velocity_y() * c1.mass() + c2.velocity_y() * c2.mass()) / m;

        // positions and velocities of the components in the CM frame
        let x1 = c1.position_x() - x_cm;
        let y1 = c1.position_y() - y_cm;
        let x2 = c2.position_x() - x_cm;
        let y2 = c2.position_y() - y_cm;
        let r1 = f64::sqrt(x1 * x1 + y1 * y1);
        let r2 = f64::sqrt(x2 * x2 + y2 * y2);
        let vx1 = c1.velocity_x() - vx_cm;
        let vy1 = c1.velocity_y() - vy_cm;
        let vx2 = c2.velocity_x() - vx_cm;
        let vy2 = c2.velocity_y() - vy_cm;
        let vf1 = -vx1 * y1 / r1 + vy1 * x1 / r1;
        let vf2 = -vx2 * y2 / r2 + vy2 * x2 / r2;
        let v1 = f64::sqrt(vx1 * vx1 + vy1 * vy1);

        // energy and angular momentum (t := kinetic energy, l := angular momentum, h := total energy)
        let t1 = 0.5 * m1 * (vx1 * vx1 + vy1 * vy1);
        let t2 = 0.5 * m2 * (vx2 * vx2 + vy2 * vy2);
        let l1 = m1 * r1 * vf1;
        let l2 = m2 * r2 * vf2;
        let r = r1 + r2;
        let l = l1 + l2;
        let h = t1 + t2 - G * m1 * m2 / r;

        if h >= 0.0 {
            return Err(UnboundOrbitalState(self))
        }

        // semi-major, semi-minor axes; eccentricity, apsides
        let a = -0.5 * G * m1 * m2 / h;
        let b = f64::sqrt(-0.5 * l * l / h * (m1 + m2) / (m1 * m2));
        let e = f64::sqrt(clamp_between_zero_and_one(1.0 - b * b / a / a));
        let omega = f64::sqrt(G * m / a / a / a);

        // semi-major and semi-minor axes of the primary
        let a1 = a * q / (1.0 + q);
        let b1 = b * q / (1.0 + q);

        // cos of nu and f: phase angle and true anomaly
        let cn = if e == 0.0 { x1 / r1 } else { (1.0 - r1 / a1) / e };
        let cf = a1 / r1 * (cn - e);

        // sin of nu and f
        let sn = if e == 0.0 { y1 / r1 } else { (vx1 * x1 + vy1 * y1) / (e * v1 * r1) * f64::sqrt(1.0 - e * e * cn * cn) };
        let sf = (b1 / r1) * sn;

        // cos and sin of eccentric anomaly
        let ck = (e + cf)                    / (1.0 + e * cf);
        let sk = f64::sqrt(1.0 - e * e) * sf / (1.0 + e * cf);

        // mean anomaly and tau
        let k = f64::atan2(sk, ck);
        let n = k - e * sk;
        let tau = t - n / omega;

        // cartesian components of semi-major axis, and the argument of periapse
        let ax = (cn - e) * x1 + sn * f64::sqrt(1.0 - e * e) * y1;
        let ay = (cn - e) * y1 - sn * f64::sqrt(1.0 - e * e) * x1;
        let pomega = f64::atan2(ay, ax);

        // final result
        let elements    = OrbitalElements(a, m, q, e);
        let orientation = OrbitalOrientation(x_cm, y_cm, vx_cm, vy_cm, pomega, tau);

        Ok(OrbitalParameters(elements, orientation))
    }
}




// ============================================================================
impl OrbitalElements {

    pub fn omega(self) -> f64 {
        let m = self.total_mass();
        let a = self.semimajor_axis();
        return f64::sqrt(G * m / a / a / a);
    }

    pub fn period(self) -> f64 {
        return 2.0 * PI / self.omega();
    }

    pub fn angular_momentum(self) -> f64 {
        let a = self.semimajor_axis();
        let m = self.total_mass();
        let q = self.mass_ratio();
        let e = self.eccentricity();
        let m1 = m / (1.0 + q);
        let m2 = m - m1;
        return m1 * m2 / m * f64::sqrt(G * m * a * (1.0 - e * e));
    }

    pub fn eccentric_anomaly(self, t: f64) -> f64 {
        let e = self.eccentricity();
        let n = self.omega() * t;              // n := mean anomaly M
        let f = |k| k   - e * f64::sin(k) - n; // k := eccentric anomaly E
        let g = |k| 1.0 - e * f64::cos(k);
        return solve_newton_rapheson(f, g, n);
    }

    pub fn orbital_state_from_eccentric_anomaly(self, eccentric_anomaly: f64) -> OrbitalState {
        let a   = self.semimajor_axis();
        let m   = self.total_mass();
        let q   = self.mass_ratio();
        let e   = self.eccentricity();
        let w   = self.omega();
        let m1  = m / (1.0 + q);
        let m2  = m - m1;
        let ck  = eccentric_anomaly.cos();
        let sk  = eccentric_anomaly.sin();
        let x1  = -a * q / (1.0 + q) * (e - ck);
        let y1  =  a * q / (1.0 + q) * (    sk) * (1.0 - e * e).sqrt();
        let x2  = -x1 / q;
        let y2  = -y1 / q;
        let vx1 = -a * q / (1.0 + q) * w / (1.0 - e * ck) * sk;
        let vy1 =  a * q / (1.0 + q) * w / (1.0 - e * ck) * ck * (1.0 - e * e).sqrt();
        let vx2 = -vx1 / q;
        let vy2 = -vy1 / q;
        let c1  = PointMass(m1, x1, y1, vx1, vy1);
        let c2  = PointMass(m2, x2, y2, vx2, vy2);
        return OrbitalState(c1, c2);
    }

    pub fn orbital_state_from_time(self, t: f64) -> OrbitalState {
        self.orbital_state_from_eccentric_anomaly(self.eccentric_anomaly(t))
    }

    /**
     * Generate the orbital state vector for the given orbital elements and
     * orientation.
     *
     * * t  -    The time
     * * o  -    The orbital orientation
     */
    pub fn orbital_state_from_time_and_orientation(self, t: f64, o: OrbitalOrientation) -> OrbitalState {
        self.orbital_state_from_time(t - o.periapse_time()).transform(o)
    }

    /**
     * Return the new orbital elements if a particle with these orbital elements
     * is perturbed by the given masses and momenta.
     *
     * * t    -    The time
     * * dm1  -    Mass added to the primary
     * * dm2  -    Mass added to the secondary
     * * dpx1 -    Force (x) added to the primary
     * * dpx2 -    Force (x) added to the secondary
     * * dpy1 -    Force (y) added to the primary
     * * dpy2 -    Force (y) added to the secondary
     */
    pub fn perturb(self, t: f64, dm1: f64, dm2: f64, dpx1: f64, dpx2: f64, dpy1: f64, dpy2: f64) -> Result<Self, UnboundOrbitalState> {
        let s0 = self.orbital_state_from_time(t);
        let s1 = s0.perturb(dm1, dm2, dpx1, dpx2, dpy1, dpy2);
        Ok(s1.recover_orbital_parameters(t)?.0)
    }
}




// ============================================================================
impl std::ops::Add<OrbitalElements> for OrbitalElements { type Output = Self; fn add(self, u: OrbitalElements) -> OrbitalElements { OrbitalElements(self.0 + u.0, self.1 + u.1, self.2 + u.2, self.3 + u.3) } }
impl std::ops::Sub<OrbitalElements> for OrbitalElements { type Output = Self; fn sub(self, u: OrbitalElements) -> OrbitalElements { OrbitalElements(self.0 - u.0, self.1 - u.1, self.2 - u.2, self.3 - u.3) } }
impl std::ops::Mul<f64> for OrbitalElements { type Output = OrbitalElements; fn mul(self, a: f64) -> OrbitalElements { OrbitalElements(self.0 * a, self.1 * a, self.2 * a, self.3 * a) } }
impl std::ops::Div<f64> for OrbitalElements { type Output = OrbitalElements; fn div(self, a: f64) -> OrbitalElements { OrbitalElements(self.0 / a, self.1 / a, self.2 / a, self.3 / a) } }


// ============================================================================
impl std::ops::Add<OrbitalOrientation> for OrbitalOrientation { type Output = Self; fn add(self, u: OrbitalOrientation) -> OrbitalOrientation { OrbitalOrientation(self.0 + u.0, self.1 + u.1, self.2 + u.2, self.3 + u.3, self.4 + u.4, self.5 + u.5) } }
impl std::ops::Sub<OrbitalOrientation> for OrbitalOrientation { type Output = Self; fn sub(self, u: OrbitalOrientation) -> OrbitalOrientation { OrbitalOrientation(self.0 - u.0, self.1 - u.1, self.2 - u.2, self.3 - u.3, self.4 - u.4, self.5 - u.5) } }
impl std::ops::Mul<f64> for OrbitalOrientation { type Output = OrbitalOrientation; fn mul(self, a: f64) -> OrbitalOrientation { OrbitalOrientation(self.0 * a, self.1 * a, self.2 * a, self.3 * a, self.4 * a, self.5 * a) } }
impl std::ops::Div<f64> for OrbitalOrientation { type Output = OrbitalOrientation; fn div(self, a: f64) -> OrbitalOrientation { OrbitalOrientation(self.0 / a, self.1 / a, self.2 / a, self.3 / a, self.4 / a, self.5 / a) } }


// ============================================================================
impl std::ops::Add<OrbitalParameters> for OrbitalParameters { type Output = Self; fn add(self, u: OrbitalParameters) -> OrbitalParameters { OrbitalParameters(self.0 + u.0, self.1 + u.1) } }
impl std::ops::Sub<OrbitalParameters> for OrbitalParameters { type Output = Self; fn sub(self, u: OrbitalParameters) -> OrbitalParameters { OrbitalParameters(self.0 - u.0, self.1 - u.1) } }
impl std::ops::Mul<f64> for OrbitalParameters { type Output = OrbitalParameters; fn mul(self, a: f64) -> OrbitalParameters { OrbitalParameters(self.0 * a, self.1 * a) } }
impl std::ops::Div<f64> for OrbitalParameters { type Output = OrbitalParameters; fn div(self, a: f64) -> OrbitalParameters { OrbitalParameters(self.0 / a, self.1 / a) } }


// ============================================================================
impl OrbitalElements     { pub fn small(self, e: f64) -> bool { self.0.abs() < e && self.1.abs() < e && self.2.abs() < e && self.3.abs() < e } }
impl OrbitalOrientation  { pub fn small(self, e: f64) -> bool { self.0.abs() < e && self.1.abs() < e && self.2.abs() < e && self.3.abs() < e && self.4.abs() < e && self.5.abs() < e } }
impl OrbitalParameters   { pub fn small(self, e: f64) -> bool { self.0.small(e) && self.1.small(e) } }




// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn panic_unless_recovery_is_accurate(t: f64, elements: OrbitalElements, orientation: OrbitalOrientation) {
        let parameters = OrbitalParameters(elements, orientation);
        let state = elements.orbital_state_from_time_and_orientation(t, orientation);
        let recovered_parameters = state.recover_orbital_parameters(t).unwrap();
        assert!((parameters - recovered_parameters).small(1e-12));
    }

    #[test]
    fn can_recover_parameters_for_standard_oriented_orbits() {
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(1.0, 1.0, 1.0, 0.1), OrbitalOrientation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(0.9, 1.0, 1.0, 0.1), OrbitalOrientation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(1.0, 0.9, 1.0, 0.1), OrbitalOrientation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(1.0, 1.0, 0.9, 0.1), OrbitalOrientation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(1.0, 1.0, 1.0, 0.9), OrbitalOrientation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn can_recover_parameters_for_nonstandard_oriented_orbits() {
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(1.0, 1.0, 1.0, 0.1), OrbitalOrientation(0.0, 0.0, 0.0, 0.0, 0.0, 0.1));
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(0.9, 1.0, 1.0, 0.1), OrbitalOrientation(0.0, 0.0, 0.0, 0.0, 0.1, 0.0));
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(1.0, 0.9, 1.0, 0.1), OrbitalOrientation(0.0, 0.0, 0.0, 0.1, 0.0, 0.0));
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(1.0, 1.0, 0.9, 0.1), OrbitalOrientation(0.0, 0.0, 0.1, 0.0, 0.0, 0.0));
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(1.0, 1.0, 1.0, 0.9), OrbitalOrientation(0.0, 0.1, 0.0, 0.0, 0.0, 0.0));
        panic_unless_recovery_is_accurate(0.3, OrbitalElements(1.0, 1.0, 1.0, 0.1), OrbitalOrientation(0.1, 0.0, 0.0, 0.0, 0.0, 0.0));
    }
}
