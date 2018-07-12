/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

std::ostream& operator<< (std::ostream& stream, const Particle& p)
{
	stream << "Particle(" << p.id << ") x=" << p.x << " y=" << p.y << " theta=" << p.theta << " weight=" << p.weight << "\n";
	for (int i = 0; i < p.associations.size(); ++i) {
		stream << "   " << p.associations[i] << " x=" << p.sense_x[i] << " y=" << p.sense_y[i] << "\n";
	}
	return stream;
}

std::ostream& operator<< (std::ostream& stream, const vector<Particle>& pv)
{
	for (auto& p : pv) {
		stream << p;
	}
	return stream;
}

std::ostream& operator<< (std::ostream& stream, const LandmarkObs& obs)
{
	stream << "LandmarkObs(" << obs.id << ") x=" << obs.x << " y=" << obs.y << "\n";
	return stream;
}

std::ostream& operator<< (std::ostream& stream, const vector<LandmarkObs>& ov)
{
	for (auto& obs : ov) {
		stream << obs;
	}
	return stream;
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 100;
	weights.reserve(num_particles);
	particles.reserve(num_particles);

	normal_distribution<double> x_gauss(x, std[0]);
	normal_distribution<double> y_gauss(y, std[1]);
	normal_distribution<double> theta_gauss(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		particles.emplace_back(
			i, x_gauss(random_engine),
			y_gauss(random_engine),
			theta_gauss(random_engine), 1.0);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	double v_per_yaw_rate = velocity / yaw_rate;
	double phi = yaw_rate * delta_t;

	normal_distribution<double> nd_x(0, std_pos[0]);
	normal_distribution<double> nd_y(0, std_pos[1]);
	normal_distribution<double> nd_theta(0, std_pos[2]);

	// Eqs from Lesson 12, Yaw Rate and Velocity
	for (auto& p : particles) {
		double x = p.x + v_per_yaw_rate * (sin(p.theta + phi) - sin(p.theta));
		double y = p.y + v_per_yaw_rate * (cos(p.theta) - cos(p.theta + phi));
		double t = p.theta + phi;
		p.x = nd_x(random_engine) + x;
		p.y = nd_y(random_engine) + y;
		p.theta = nd_theta(random_engine) + t;
	}
}

std::vector<int> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	if (predicted.empty() || observations.empty()) {
		return vector<int>();
	}

	vector<int> out_ids;
	out_ids.reserve(observations.size());

	auto dist2 = [](const LandmarkObs& a, const LandmarkObs& b) -> double
	{
		double dx = a.x - b.x;
		double dy = a.y - b.y;
		return dx * dx + dy * dy;
	};

	for (auto& obs : observations) {
		auto el = min_element(predicted.begin(), predicted.end(),
			[&obs, &dist2](const LandmarkObs& a, const LandmarkObs& b)
			{
				return dist2(obs, a) < dist2(obs, b);
			});
		obs.id = el->id;
		out_ids.push_back(distance(predicted.begin(), el));
	}
	return out_ids;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	auto toMapCoordinates = [](const LandmarkObs& obs, const Particle& p) -> LandmarkObs
	{
		LandmarkObs new_obs;
		new_obs.id = obs.id;
		new_obs.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
		new_obs.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
		return new_obs;
	};

	auto dist2 = [](const Map::single_landmark_s& a, const Particle& b) -> double
	{
		double dx = a.x_f - b.x;
		double dy = a.y_f - b.y;
		return dx * dx + dy * dy;
	};

	double sx = 1.0 / (std_landmark[0] * std_landmark[0]);
	double sy = 1.0 / (std_landmark[1] * std_landmark[1]);

	auto score = [sx, sy](const LandmarkObs& a, const LandmarkObs& b) -> double
	{
		double dx = a.x - b.x;
		double dy = a.y - b.y;
		// since all the weights are to be normalized, no need for a proper
		// multivariate normal pdf evaluation
		return exp(-0.5 * (sx * dx * dx + sy * dy * dy));
	};

	double sensor_range_square = sensor_range * sensor_range;

	double weight_sum = 0.0;

	for (auto& p : particles) {
		vector<LandmarkObs> in_predicted_range;

		// predict the landmarks for the particle, using the map
		for (const Map::single_landmark_s& lm : map_landmarks.landmark_list) {
			if (dist2(lm, p) < sensor_range_square) {
				in_predicted_range.push_back(LandmarkObs({
					lm.id_i, (double)lm.x_f, (double)lm.y_f}));
			}
		}

		// transform the observations from the particle coordinate system
		// to the map coordinate system
		vector<LandmarkObs> observations_transf(observations);
		for (auto& obs : observations_transf) {
			obs = toMapCoordinates(obs, p);
		}

		// select the closest observation for every predicted landmark
		// so after calling dataAssociation, the closest observation to
		// in_predicted_range[i] is observations_transf[i]
		auto pred_indices = dataAssociation(in_predicted_range, observations_transf);

		// calculate the weight for all the 
		// predicted landmark - observation pairs
		double weight = 0.0;
		for (int i = 0; i < observations_transf.size(); ++i) {
			if (i < pred_indices.size()) {
				int pred_id = pred_indices[i];
				weight += score(in_predicted_range[pred_id], observations_transf[i]);
			}
		}
		if (!observations_transf.empty()) {
			weight /= (double)observations_transf.size();
			p.weight = weight;
			weight_sum += weight;
		}

		// store the gathered information
		p.associations.clear();
		p.sense_x.clear();
		p.sense_y.clear();
		for (auto& obs : observations_transf) {
			p.associations.push_back(obs.id);
			p.sense_x.push_back(obs.x);
			p.sense_y.push_back(obs.y);
		};
	}

	// weight normalization
	// if (weight_sum > 10e-6) {
		// double scale = 1.0 / weight_sum;
		cout << "nobs " << observations.size() << endl;
		if (!observations.empty()) {
			for (int i = 0; i < particles.size(); ++i) {
				// particles[i].weight *= scale;
				weights[i] = particles[i].weight;
			}
		}
	// }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	discrete_distribution<double> dd(weights.begin(), weights.end());
	vector<Particle> old_particles(particles);
	for (int i = 0; i < particles.size(); ++i) {
		int k = dd(random_engine);
		cout << k << ", ";
		particles[i] = old_particles[k];
	}
	cout << endl;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
