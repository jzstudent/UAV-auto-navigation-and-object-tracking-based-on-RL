#ifndef air_DroneCommon_hpp
#define air_DroneCommon_hpp

#include "common/Common.hpp"
#include "common/CommonStructs.hpp"
#include "physics/Kinematics.hpp"

namespace msr {  namespace airlib {

enum class DrivetrainType {
    MaxDegreeOfFreedom = 0,
    ForwardOnly
};

enum class LandedState : uint {
    Landed = 0,
    Flying = 1
};

//Yaw mode specifies if yaw should be set as angle or angular velocity around the center of drone
struct YawMode {
    bool is_rate = true;
    float yaw_or_rate = 0.0f;

    YawMode()
    {}

    YawMode(bool is_rate_val, float yaw_or_rate_val)
    {
        is_rate = is_rate_val;
        yaw_or_rate = yaw_or_rate_val;
    }

    static YawMode Zero()
    {
        return YawMode(true, 0);
    }

    void setZeroRate()
    {
        is_rate = true;
        yaw_or_rate = 0;
    }
};

//properties of vehicle
struct MultirotorApiParams {
    MultirotorApiParams() {};
    //what is the breaking distance for given velocity?
    //Below is just proportionality constant to convert from velocity to breaking distance
    float vel_to_breaking_dist = 0.5f;   //ideally this should be 2X for very high speed but for testing we are keeping it 0.5
    float min_breaking_dist = 1; //min breaking distance
    float max_breaking_dist = 3; //min breaking distance
    float breaking_vel = 1.0f;
    float min_vel_for_breaking = 3;

    //what is the differential positional accuracy of cur_loc?
    //this is not same as GPS accuracy because translational errors
    //usually cancel out. Typically this would be 0.2m or less
    float distance_accuracy = 0.1f;

    //what is the minimum clearance from obstacles?
    float obs_clearance = 2;

    //what is the +/-window we should check on obstacle map?
    //for example 2 means check from ticks -2 to 2
    int obs_window = 0;
};

struct TripStats{
    float state_of_charge;
    float voltage;
    float energy_consumed;
    float flight_time;
    float distance_traveled;
    int collision_count;  
    
    TripStats()
    {}
    
    TripStats(float state_of_charge_val,
                float voltage_val,
                float energy_consumed_val,
                float flight_time_val,
                float distance_traveled_val,
                int collision_count_val):
        state_of_charge(state_of_charge_val), 
        voltage(voltage_val), 
        energy_consumed(energy_consumed_val), 
        flight_time(flight_time_val), 
        distance_traveled(distance_traveled_val),
        collision_count(collision_count_val)
    {
    }

};

struct MultirotorState {
    CollisionInfo collision;
    Kinematics::State kinematics_estimated;
    GeoPoint gps_location;
    uint64_t timestamp;
    LandedState landed_state;
    RCData rc_data;
    TripStats trip_stats;

    MultirotorState()
    {}
    MultirotorState(const CollisionInfo& collision_val, const Kinematics::State& kinematics_estimated_val, 
        const GeoPoint& gps_location_val, uint64_t timestamp_val,
        LandedState landed_state_val, const RCData& rc_data_val, const TripStats& trip_stats)
        : collision(collision_val), kinematics_estimated(kinematics_estimated_val),
        gps_location(gps_location_val), timestamp(timestamp_val),
        landed_state(landed_state_val), rc_data(rc_data_val), trip_stats(trip_stats)
    {
    }

    //shortcuts
    const Vector3r& getPosition() const
    {
        return kinematics_estimated.pose.position;
    }
    const Quaternionr& getOrientation() const
    {
        return kinematics_estimated.pose.orientation;
    }
    const TripStats& getTripStats() const{
        return trip_stats;
    }
};

}} //namespace
#endif
