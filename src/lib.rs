use std::f64::consts::{PI};
static G: f64 = 1.0;




// ============================================================================
#[derive(Debug)]
pub struct UboundOrbitalState {}

impl std::fmt::Display for UboundOrbitalState
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result
    {
        write!(f, "orbital state vector does not correspond to a bound orbit")
    }
}

impl std::error::Error for UboundOrbitalState {}




// ============================================================================
#[derive(Clone, Copy)]
pub struct PointMass(f64, f64, f64, f64, f64);

#[derive(Clone, Copy)]
pub struct OrbitalState(PointMass, PointMass);

#[derive(Clone, Copy)]
pub struct OrbitalElements(f64, f64, f64, f64);

#[derive(Clone, Copy)]
pub struct OrbitalOrientation(f64, f64, f64, f64, f64, f64);

#[derive(Clone, Copy)]
pub struct OrbitalParameters(OrbitalElements, OrbitalOrientation);




// ============================================================================
impl PointMass
{
    pub fn mass          (self) -> f64 { self.0 }
    pub fn position_x    (self) -> f64 { self.1 }
    pub fn position_y    (self) -> f64 { self.2 }
    pub fn velocity_x    (self) -> f64 { self.3 }
    pub fn velocity_y    (self) -> f64 { self.4 }
}




// ============================================================================
impl OrbitalElements
{
    pub fn semimajor_axis(self) -> f64 { self.0 }
    pub fn total_mass    (self) -> f64 { self.1 }
    pub fn mass_ratio    (self) -> f64 { self.2 }
    pub fn eccentricity  (self) -> f64 { self.3 }
}




// ============================================================================
impl OrbitalOrientation
{
    pub fn cm_position_x    (self) -> f64 { self.0 }
    pub fn cm_position_y    (self) -> f64 { self.1 }
    pub fn cm_velocity_x    (self) -> f64 { self.2 }
    pub fn cm_velocity_y    (self) -> f64 { self.3 }
    pub fn periapse_argument(self) -> f64 { self.4 }
    pub fn periapse_time    (self) -> f64 { self.5 }
}




// ============================================================================
fn solve_newton_rapheson<F: Fn(f64) -> f64, G: Fn(f64) -> f64>(f: F, g: G, mut x: f64) -> f64
{
    while f64::abs(f(x)) > 1e-12
    {
        x -= f(x) / g(x);
    }
    return x;
}

fn clamp_between_zero_and_one(x: f64) -> f64
{
    f64::min(1.0, f64::max(0.0, x))
}




// ============================================================================
impl PointMass
{
    pub fn kinetic_energy(self) -> f64
    {
        let vx = self.velocity_x();
        let vy = self.velocity_y();
        return 0.5 * self.mass() * (vx * vx + vy * vy);
    }

    pub fn angular_momentum(self) -> f64
    {
        let x  = self.position_x();
        let y  = self.position_y();
        let vx = self.velocity_x();
        let vy = self.velocity_y();
        return self.mass() * (x * vy - y * vx);
    }

    pub fn gravitational_potential(self, x: f64, y: f64, softening_length: f64) -> f64
    {
        let dx = x - self.position_x();
        let dy = y - self.position_y();
        let r2 = dx * dx + dy * dy;
        let s2 = softening_length.powi(2);
        return -G * self.mass() / (r2 + s2).sqrt();
    }

    pub fn gravitational_acceleration(self, x: f64, y: f64, softening_length: f64) -> [f64; 2]
    {
        let dx = x - self.position_x();
        let dy = y - self.position_y();
        let r2 = dx * dx + dy * dy;
        let s2 = softening_length.powi(2);
        let ax = -G * self.mass() / (r2 + s2).powf(1.5) * dx;
        let ay = -G * self.mass() / (r2 + s2).powf(1.5) * dy;
        return [ax, ay];
    }

    pub fn perturb_mass(self, dm: f64) -> PointMass
    {
        let mut p = self;
        p.0 += dm;
        return p;
    }

    pub fn perturb_momentum(self, dpx: f64, dpy: f64) -> PointMass
    {
        let mut p = self;
        p.3 += dpx / self.mass();
        p.4 += dpy / self.mass();
        return p;
    }

    pub fn perturb_mass_and_momentum(self, dm: f64, dpx: f64, dpy: f64) -> PointMass
    {
        let mut p = self;
        p.0 += dm;
        p.3 += dpx / self.mass();
        p.4 += dpy / self.mass();
        return p;
    }
}




// ============================================================================
impl OrbitalState
{
    pub fn total_mass(self) -> f64
    {
        return self.0.mass() + self.1.mass();
    }

    pub fn mass_ratio(self) -> f64
    {
        return self.1.mass() / self.0.mass();
    }

    pub fn separation(self) -> f64
    {
        let x1 = self.0.position_x();
        let y1 = self.0.position_y();
        let x2 = self.1.position_x();
        let y2 = self.1.position_y();
        return f64::sqrt((x2 - x1).powi(2) + (y2 - y1).powi(2));
    }

    pub fn kinetic_energy(self) -> f64
    {
        return self.0.kinetic_energy() + self.1.kinetic_energy();
    }

    pub fn angular_momentum(self) -> f64
    {
        return self.0.angular_momentum() + self.1.angular_momentum();
    }

    pub fn potential(self, x: f64, y: f64, softening_length: f64) -> f64
    {
        return self.0.potential(x, y, softening_length) + self.1.potential(x, y, softening_length);
    }

    pub fn total_energy(self) -> f64
    {
        return self.kinetic_energy() - G * self.0.mass() * self.1.mass() / self.separation();
    }

    /**
     * Transform this orbital state vector so that it has the new orientation.
     * This function rotates the position and velocity vectors according to the
     * argument of periapse, and translates them according to the center-of-mass
     * position and velocity. Note that the time-of-last-periapse-passage is
     * technically part of the orbital orientation, but is not used by this
     * function, as that would change the intrinsic orbital phase.
     *
     * # Arguments
     * * o      - The orbital orientation to transform to
     */
    pub fn transform(self, o: OrbitalOrientation) -> OrbitalState
    {
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

        return OrbitalState(c1, c2);
    }

    /**
     * Rotate an orbital state vector by the given angle: positive angle means
     * that the argument of periapse moves counter-clockwise, in other words
     * this function rotates the binary, not the coordinates.
     *
     * * angle  - The angle to rotate by
     *
     */
    pub fn rotate(self, angle: f64) -> OrbitalState
    {
        let orientation = OrbitalOrientation(0.0, 0.0, 0.0, 0.0, angle, 0.0);
        return self.transform(orientation);
    }

    pub fn recover_orbital_parameters(self, t: f64) -> Result<OrbitalParameters, UboundOrbitalState>
    {
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
            return Err(UboundOrbitalState{});
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
impl OrbitalElements
{
    pub fn omega(self) -> f64
    {
        let m = self.total_mass();
        let a = self.semimajor_axis();
        return f64::sqrt(G * m / a / a / a);
    }

    pub fn period(self) -> f64
    {
        return 2.0 * PI / self.omega();
    }

    pub fn angular_momentum(self) -> f64
    {
        let a = self.semimajor_axis();
        let m = self.total_mass();
        let q = self.mass_ratio();
        let e = self.eccentricity();
        let m1 = m / (1.0 + q);
        let m2 = m - m1;
        return m1 * m2 / m * f64::sqrt(G * m * a * (1.0 - e * e));
    }

    pub fn eccentric_anomaly(self, t: f64) -> f64
    {
        let e = self.eccentricity();
        let n = self.omega() * t;              // n := mean anomaly M
        let f = |k| k   - e * f64::sin(k) - n; // k := eccentric anomaly E
        let g = |k| 1.0 - e * f64::cos(k);
        return solve_newton_rapheson(f, g, n);
    }

    pub fn orbital_state_from_eccentric_anomaly(self, eccentric_anomaly: f64) -> OrbitalState
    {
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

    /**
     * Generate the orbital state vector for the given orbital elements and
     * orientation.
     *
     * * o  -    The orbital orientation
     * * t  -    The time
     */
    pub fn orbital_state_from_time_and_orientation(self, o: OrbitalOrientation, t: f64) -> OrbitalState
    {
        self.orbital_state_from_eccentric_anomaly(self.eccentric_anomaly(t - o.periapse_time())).transform(o)
    }
}
