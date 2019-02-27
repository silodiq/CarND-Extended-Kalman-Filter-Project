#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

// Constructor
FusionEKF::FusionEKF() {
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	Hj_      = MatrixXd(3, 4);

	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
		        0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
		        0, 0.0009, 0,
		        0, 0, 0.09;

	// P Initialization
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ << 1, 0, 0,    0,
			   0, 1, 0,    0,
			   0, 0, 1000, 0,
			   0, 0, 0,    1000;

	H_laser_ << 1, 0, 0, 0,
				0, 1, 0, 0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {
}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

	// ============= first measurement =============
	if (false == is_initialized_) {
		
		ekf_.x_ = VectorXd(4);
		ekf_.x_ << 1, 1, 1, 1;

		// state transition matrix initialization
		ekf_.F_ = MatrixXd(4, 4);
		ekf_.F_ <<  1, 0, 1, 0,
					0, 1, 0, 1,
					0, 0, 1, 0,
					0, 0, 0, 1;

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			cout << "EKF : First radar measurement !" << endl;
      
			// convert from polar to cartesian coordinates      
			double rho = measurement_pack.raw_measurements_[0]; // range
  			double phi = measurement_pack.raw_measurements_[1]; // bearing
  			double rho_dot = measurement_pack.raw_measurements_[2]; // velocity of rho  		
  			double x = rho * cos(phi);		
			(x < 0.0001) ? (x = 0.0001) : (x = x);
  			double y = rho * sin(phi);		
			(y < 0.0001) ? (y = 0.0001) : (y = y);
  			double vx = rho_dot * cos(phi);
  			double vy = rho_dot * sin(phi);
			// state initialization
			ekf_.x_ << x, y, vx , vy;
		}
		else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
			//state initialization
			cout << "EKF : First lidar measurement !" << endl;
			ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
		}

		// save timestamp as previous stamp
		previous_timestamp_ = measurement_pack.timestamp_ ;
		// indicate first measurement has been done
		is_initialized_ = true;
		return;
	}

	// ============= prediction =============  
    
	// get delta time 
	double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = measurement_pack.timestamp_;

	
	// Update the state transition matrix F according to the new elapsed time.	
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;

	// Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	double noise_ax = 9.0;
	double noise_ay = 9.0;

	double dt_2 = dt * dt;    //dt^2
	double dt_3 = dt_2 * dt;  //dt^3
	double dt_4 = dt_3 * dt;  //dt^4
	double dt_4_4 = dt_4 / 4; //dt^4/4
	double dt_3_2 = dt_3 / 2; //dt^3/2

	// Update the process noise covariance matrix.
	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.Q_ << dt_4_4 * noise_ax, 0,                 dt_3_2 * noise_ax, 0,
			   0,                 dt_4_4 * noise_ay, 0,                 dt_3_2 * noise_ay,
			   dt_3_2 * noise_ax, 0,                 dt_2 * noise_ax,   0,
 			   0,                 dt_3_2 * noise_ay, 0,                 dt_2 * noise_ay;


	ekf_.Predict();


	// ============= update =============    
    
	// Use the sensor type to perform the update step.
	// Update the state and covariance matrices.
	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
		ekf_.R_ = R_radar_;
		ekf_.UpdateEKF(measurement_pack.raw_measurements_);
	} else {
		// Laser updates
		ekf_.H_ = H_laser_;
		ekf_.R_ = R_laser_;
		ekf_.Update(measurement_pack.raw_measurements_);
	}

	// print the output
	cout << "x_ = " << ekf_.x_ << endl;
	cout << "P_ = " << ekf_.P_ << endl;
}
