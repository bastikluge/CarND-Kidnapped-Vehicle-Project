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

  // Create normal (Gaussian) distribution for x, y theta.
  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  num_particles = 1000;
  weights.resize(num_particles);
  particles.resize(num_particles);
  for ( int i=0; i<num_particles; i++ )
  {
    // set weight to 1.0
    weights[i] = 1.0;

    // initialize particle according to normal distributions
    Particle &p = particles[i];
    p.id     = i;
    p.x      = dist_x(gen);
    p.y      = dist_y(gen);
    p.theta  = dist_theta(gen);
    while ( p.theta >  M_PI ) p.theta -= M_PI;
    while ( p.theta < -M_PI ) p.theta += M_PI;
    p.weight = weights[i];
    p.associations.clear();
    p.sense_x.clear();
    p.sense_y.clear();
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  // Create normal (Gaussian) distribution for x, y theta.
  default_random_engine gen;
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  if ( abs(yaw_rate) < 0.00000001 )
  {
    for ( int i=0; i<num_particles; i++ )
    {
      // exact process model
      Particle &p = particles[i];
      //if ( i < 10 ) cout << "moved from (" << p.x << ", " << p.y << ", " << p.theta << ")" << endl;
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
      // p.theta = p.theta;

      // process noise
      p.x     += dist_x(gen);
      p.y     += dist_y(gen);
      p.theta += dist_theta(gen);
      while ( p.theta >  M_PI ) p.theta -= M_PI;
      while ( p.theta < -M_PI ) p.theta += M_PI;
      //if ( i < 10 ) cout << "    ... to (" << p.x << ", " << p.y << ", " << p.theta << ")" << endl;
    }
  }
  else
  {
    for ( int i=0; i<num_particles; i++ )
    {
      // exact process model
      Particle &p = particles[i];
      //if ( i < 10 ) cout << "moved from (" << p.x << ", " << p.y << ", " << p.theta << ")" << endl;
      p.x     += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      p.x     += velocity / yaw_rate * (cos(p.theta)                      - cos(p.theta + yaw_rate * delta_t));
      p.theta += yaw_rate * delta_t;

      // process noise
      p.x     += dist_x(gen);
      p.y     += dist_y(gen);
      p.theta += dist_theta(gen);
      while ( p.theta >  M_PI ) p.theta -= M_PI;
      while ( p.theta < -M_PI ) p.theta += M_PI;
      //if ( i < 10 ) cout << "    ... to (" << p.x << ", " << p.y << ", " << p.theta << ")" << endl;
    }
  }
}

void ParticleFilter::dataAssociation(const std::vector<LandmarkObs>& predicted, const std::vector<LandmarkObs>& observations, std::vector<int> &pred_to_obs_index) {

  pred_to_obs_index.clear();
  pred_to_obs_index.resize(predicted.size(), -1);

  for ( unsigned p=0; p<predicted.size(); p++ )
  {
    double min_dist = 10000000.0;
    for ( unsigned o=0; o<observations.size(); o++ )
    {
      double cur_dist = (predicted[p].x - observations[o].x) * (predicted[p].x - observations[o].x)
                      + (predicted[p].y - observations[o].y) * (predicted[p].y - observations[o].y);
      if ( cur_dist < min_dist )
      {
        min_dist             = cur_dist;
        pred_to_obs_index[p] = o;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
  const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

  // loop over all particles
  for ( int i=0; i<num_particles; i++ )
  {
    // predict measurements for this particle
    Particle &p = particles[i];
    std::vector<LandmarkObs> predicted;
    unsigned num_landmarks = map_landmarks.landmark_list.size();
    predicted.reserve(num_landmarks);
    for ( unsigned n=0; n<num_landmarks; n++ )
    {
      const Map::single_landmark_s &landmark = map_landmarks.landmark_list[n];

      // calculate translation in world coordinates
      double dx_world = landmark.x_f - p.x;
      double dy_world = landmark.y_f - p.y;

      // add only landmarks in sensor range
      if ( sqrt(dx_world * dx_world + dy_world * dy_world) < sensor_range )
      {
        LandmarkObs pred;
        pred.id = landmark.id_i;

        // rotate translation according to particle orientation
        pred.x =  cos(p.theta) * dx_world + sin(p.theta) * dy_world;
        pred.y = -sin(p.theta) * dx_world + cos(p.theta) * dy_world;

        predicted.push_back(pred);
      }
    }

    // associate predicted measurements with actual measurements
    std::vector<int> pred_to_obs_index;
    dataAssociation(predicted, observations, pred_to_obs_index);

    // update weight
    p.weight = 1.0;
    for ( unsigned n=0; n<predicted.size(); n++ )
    {
      const LandmarkObs &pred = predicted[n];
      double dx(sensor_range), dy(sensor_range);
      if ( pred_to_obs_index[n] != -1 )
      {
        const LandmarkObs &obs = observations[pred_to_obs_index[n]];
        //if ( i == 0 ) cout << "Landmark association [" << pred.id << "]: pred=(" << pred.x << ", " << pred.y << "), ";
        //if ( i == 0 ) cout <<                                            "obs=(" << obs.x  << ", " << obs.y  << ")" << endl;
        dx = pred.x - obs.x;
        dy = pred.y - obs.y;
      }
      else
      {
        cout << "Landmark association [" << pred.id << "]: NONE" << endl;
      }

      double arg = (dx * dx) / (std_landmark[0] * std_landmark[0])
        + (dy * dy) / (std_landmark[1] * std_landmark[1]);
      arg *= -0.5;
      double probab = exp(arg) / (2*M_PI * std_landmark[0] * std_landmark[1]);
      //if ( i == 0 ) cout << "   ---> probab=" << probab << endl;
      p.weight *= probab;
    }
    weights[i] = p.weight;
  }

  ///////////////////////////////////////////////////////////////////////
  // debug output begin
  int best_idx(-1);
  double best_weight(0.0);
  for ( int i=0; i<num_particles; i++ )
  {
    if ( particles[i].weight > best_weight )
    {
      best_idx    = i;
      best_weight = particles[i].weight;
    }
  }
  if ( best_idx != -1 )
  {
    const Particle &p = particles[best_idx];
    std::vector<LandmarkObs> predicted;
    std::vector<Map::single_landmark_s> predicted_world;
    unsigned num_landmarks = map_landmarks.landmark_list.size();
    predicted.reserve(num_landmarks);
    for ( unsigned n=0; n<num_landmarks; n++ )
    {
      const Map::single_landmark_s &landmark = map_landmarks.landmark_list[n];

      // calculate translation in world coordinates
      double dx_world = landmark.x_f - p.x;
      double dy_world = landmark.y_f - p.y;

      // add only landmarks in sensor range
      if ( sqrt(dx_world * dx_world + dy_world * dy_world) < sensor_range )
      {
        LandmarkObs pred;
        pred.id = landmark.id_i;

        // rotate translation according to particle orientation
        pred.x =  cos(p.theta) * dx_world + sin(p.theta) * dy_world;
        pred.y = -sin(p.theta) * dx_world + cos(p.theta) * dy_world;

        predicted.push_back(pred);
        predicted_world.push_back(landmark);
      }
    }

    // associate predicted measurements with actual measurements
    std::vector<int> pred_to_obs_index;
    dataAssociation(predicted, observations, pred_to_obs_index);

    cout << "Best particle: (" << p.x << ", " << p.y << ", " << p.theta << ") with weight " << p.weight << endl;

    // set associations
    std::vector<int> associations;
    std::vector<double> sense_x, sense_y;
    for ( unsigned n=0; n<predicted.size(); n++ )
    {
      if ( pred_to_obs_index[n] != -1 )
      {
        const LandmarkObs &obs = observations[pred_to_obs_index[n]];

        cout << " -> LM[" << predicted[n].id << "]: pos=(" << predicted_world[n].x_f << ", " << predicted_world[n].y_f << ")" << endl;
        cout << "       pred=(" << predicted[n].x << ", " << predicted[n].y << "), ";
        cout <<         "obs=(" << obs.x          << ", " << obs.y          << "), " << endl;
        cout << "dist=" << (predicted[n].x-obs.x)*(predicted[n].x-obs.x) + (predicted[n].y-obs.y)*(predicted[n].y-obs.y) << endl; 
        associations.push_back(predicted[n].id);
        
        // convert to world coordinates
        double dx_world = cos(p.theta) * obs.x - sin(p.theta) * obs.y;
        double dy_world = sin(p.theta) * obs.x + cos(p.theta) * obs.y;
        sense_x.push_back(p.x + dx_world);
        sense_y.push_back(p.y + dy_world);
      }
    }

    particles[best_idx] = SetAssociations(p, associations, sense_x, sense_y);
  }
  // debug output end
  ///////////////////////////////////////////////////////////////////////
}

void ParticleFilter::resample() {

  // create discrete distribution according to weights
  default_random_engine gen;
  discrete_distribution<int> dist(weights.begin(), weights.end());

  // resample
  std::vector<Particle> res_particles;
  res_particles.reserve(num_particles);
  for ( int i=0; i<num_particles; i++ )
  {
    int res_idx = dist(gen);
    //if ( i < 10 ) cout << "resampled " << res_idx << " with weight " << weights[res_idx] << endl;
    res_particles.push_back( particles[res_idx] );
  }

  // set resampled particles as particles
  particles = res_particles;
  for ( int i=0; i<num_particles; i++ )
  {
    weights[i] = particles[i].weight;
  }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

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
