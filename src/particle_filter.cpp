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

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 100;
    particles.resize(num_particles);
    weights.resize(num_particles);

    std::default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];
     
    
    // normal distributions for x,y, theta
    std::normal_distribution<double> dist_x(x, std_x);
    std::normal_distribution<double> dist_y(y, std_y);
    std::normal_distribution<double> dist_theta(theta, std_theta);
    

    for (int i = 0; i < num_particles; ++i) {
        double sample_x, sample_y, sample_theta;
        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);

        particles[i] = Particle {i , sample_x, sample_y, sample_theta, 1};
        weights[i] = 1;
        
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    std::default_random_engine gen;

    for (Particle& particle : particles)
    {
        double mean_x, mean_y, mean_theta ;
        // different equation depending on yaw_rate > 0
        if (yaw_rate < 0.00001){
            mean_x = particle.x + (velocity * delta_t * cos(particle.theta));
            mean_y = particle.y + (velocity * delta_t * sin(particle.theta));

        }
        else{
            mean_x = particle.x + (velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta)));
            mean_y = particle.y + (velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t)));
        }

        mean_theta = particle.theta + yaw_rate * delta_t;

        std::normal_distribution<double> dist_x(mean_x, std_pos[0]);
        std::normal_distribution<double> dist_y(mean_y, std_pos[1]);
        std::normal_distribution<double> dist_theta(mean_theta, std_pos[2]);
        
        // Update particle with new estimate
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.
    for (LandmarkObs& observation: observations){
        double nearest_predicted_distance = INFINITY;
        int nearest_predicted = 0;

        for (LandmarkObs& prediction : predicted){
            double distance = dist(prediction.x, prediction.y, observation.x, observation.y);
            if (distance < nearest_predicted_distance){
                nearest_predicted_distance = distance;
                nearest_predicted = prediction.id;
            }
        }

        observation.id = nearest_predicted;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html

    for (int i=0; i < num_particles; i++){

        Particle particle = particles[i];

        // tranform observations to MAP coordinate system
        std::vector<LandmarkObs> transformed_observations;

        for (LandmarkObs& observation: observations){
            LandmarkObs transformed_observation;
            transformed_observation.id = observation.id;
            transformed_observation.x = observation.x * cos(particle.theta) - observation.y * sin(particle.theta) + particle.x;
            transformed_observation.y = observation.x * sin(particle.theta) + observation.y * cos(particle.theta) + particle.y;
            transformed_observations.push_back(transformed_observation);
        }

        // Obtain predicted landmark list
        std::vector<LandmarkObs> predicted;
        for (Map::single_landmark_s& landmark : map_landmarks.landmark_list) {

          if (dist(landmark.x_f, landmark.y_f, particle.x, particle.y) <= sensor_range) {
            LandmarkObs prediction {landmark.id_i, landmark.x_f, landmark.y_f};
            predicted.push_back(prediction);
          }
        }

        // associate observations with predictions
        dataAssociation(predicted, transformed_observations);

        // update weight of particle 
        particle.weight = 1;
        // For each observation
        for (const LandmarkObs& observation : transformed_observations)
        {
        
          // Get current prediction
          Map::single_landmark_s prediction = map_landmarks.landmark_list[observation.id - 1];
          
          // Differences in x and y between measurement and prediction
          double dx = observation.x - prediction.x_f;
          double dy = observation.y - prediction.y_f;
          
          // Calculate the new weight
          double new_weight = 1 / (M_PI * 2 * std_landmark[0] * std_landmark[1]) *
            std::exp(-1 * (pow(dx, 2) / pow(std_landmark[0], 2) + pow(dy, 2) / pow(std_landmark[1], 2)));
          
          // Multiply running product of weights by the new weight
          particle.weight *= new_weight;
        }
        weights[i] = particle.weight;

    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::default_random_engine gen;
    std::discrete_distribution<int> distribution {weights.begin(), weights.end()};

    std::vector<Particle> new_particles;
    for (int i=0; i < num_particles; i++){
        int new_particle_index = distribution(gen);
        Particle new_particle = particles[new_particle_index];
        new_particles.push_back(new_particle);
    }

    particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}
