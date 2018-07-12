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

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	num_particles = 200;
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

	normal_distribution<double> nd_x(0, std_pos[0]);
	normal_distribution<double> nd_y(0, std_pos[1]);
	normal_distribution<double> nd_theta(0, std_pos[2]);

	if (abs(yaw_rate) < 10e-6) {
		double r = velocity * delta_t;
		for (auto& p : particles) {
			double x = p.x + r * cos(p.theta);
			double y = p.y + r * sin(p.theta);
			double t = p.theta;
			p.x = nd_x(random_engine) + x;
			p.y = nd_y(random_engine) + y;
			p.theta = nd_theta(random_engine) + t;
		}
	} else {
		double v_per_yaw_rate = velocity / yaw_rate;
		double phi = yaw_rate * delta_t;

		for (auto& p : particles) {
			double x = p.x + v_per_yaw_rate * (sin(p.theta + phi) - sin(p.theta));
			double y = p.y + v_per_yaw_rate * (cos(p.theta) - cos(p.theta + phi));
			double t = p.theta + phi;
			p.x = nd_x(random_engine) + x;
			p.y = nd_y(random_engine) + y;
			p.theta = nd_theta(random_engine) + t;
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

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
		if (el != predicted.end()) {
			obs.id = distance(predicted.begin(), el);
		} else {
			obs.id = -1;
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

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

	double sx = 0.5 / (std_landmark[0] * std_landmark[0]);
	double sy = 0.5 / (std_landmark[1] * std_landmark[1]);
	double gauss_normalizer = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);

	auto score = [sx, sy, gauss_normalizer](const LandmarkObs& a, const LandmarkObs& b) -> double
	{
		double dx = a.x - b.x;
		double dy = a.y - b.y;
		return exp(-(sx * dx * dx + sy * dy * dy)) * gauss_normalizer;
	};

	double sensor_range_square = sensor_range * sensor_range;

	for (auto& p : particles) {
		
		// predict the landmarks for the particle, using the map
		vector<LandmarkObs> predictions_in_range;
		for (const Map::single_landmark_s& lm : map_landmarks.landmark_list) {
			if (dist2(lm, p) <= sensor_range_square) {
				predictions_in_range.push_back(LandmarkObs({
					lm.id_i, (double)lm.x_f, (double)lm.y_f}));
			}
		}

		// transform the observations from the particle coordinate system
		// to the map coordinate system
		vector<LandmarkObs> observations_transf(observations);
		for (auto& obs : observations_transf) {
			obs = toMapCoordinates(obs, p);
		}

		// associate
		dataAssociation(predictions_in_range, observations_transf);

		// calculate the weight for all the 
		// predicted landmark - observation pairs
		double weight = 1.0;
		for (auto& obs : observations_transf) {
			if (obs.id >= 0) {
				weight *= score(predictions_in_range[obs.id], obs);
			}
		}
		p.weight = weight;

		// store the gathered information
		p.associations.clear();
		p.sense_x.clear();
		p.sense_y.clear();
		for (auto& obs : observations_transf) {
			if (obs.id >= 0) {
				p.associations.push_back(predictions_in_range[obs.id].id);
				p.sense_x.push_back(obs.x);
				p.sense_y.push_back(obs.y);
			}
		};
	}

	for (int i = 0; i < particles.size(); ++i) {
		// std::discrete_distribution will do the normalization,
		// so we can skip it here
		weights[i] = particles[i].weight;
	}
}

void ParticleFilter::resample() {

	discrete_distribution<double> dd(weights.begin(), weights.end());
	vector<Particle> old_particles(particles);
	for (int i = 0; i < particles.size(); ++i) {
		int k = dd(random_engine);
		particles[i] = old_particles[k];
	}
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
	return particle;
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
