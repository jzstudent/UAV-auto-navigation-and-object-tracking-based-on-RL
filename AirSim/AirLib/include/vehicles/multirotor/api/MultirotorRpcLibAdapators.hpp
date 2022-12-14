// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef air_MultirotorRpcLibAdapators_hpp
#define air_MultirotorRpcLibAdapators_hpp

#include "common/Common.hpp"
#include "common/CommonStructs.hpp"
#include "api/RpcLibAdapatorsBase.hpp"
#include "vehicles/multirotor/api/MultirotorCommon.hpp"
#include "vehicles/multirotor/api/MultirotorApiBase.hpp"
#include "common/ImageCaptureBase.hpp"
#include "safety/SafetyEval.hpp"

#include "common/common_utils/WindowsApisCommonPre.hpp"
#include "rpc/msgpack.hpp"
#include "common/common_utils/WindowsApisCommonPost.hpp"

namespace msr { namespace airlib_rpclib {

class MultirotorRpcLibAdapators : public RpcLibAdapatorsBase {
public:
    struct YawMode {
        bool is_rate = true;
        float yaw_or_rate = 0;
        MSGPACK_DEFINE_MAP(is_rate, yaw_or_rate);
    
        YawMode()
        {}

        YawMode(const msr::airlib::YawMode& s)
        {
            is_rate = s.is_rate;
            yaw_or_rate = s.yaw_or_rate;
        }
        msr::airlib::YawMode to() const
        {
            return msr::airlib::YawMode(is_rate, yaw_or_rate);
        }
    };

    struct TripStats {
    
        float state_of_charge = -1.0f;
        float voltage = -1.0f;
        float energy_consumed = -1.0f;
        float flight_time = -1.0;
        float distance_traveled = -1.0f;
        int collision_count = 0;

        MSGPACK_DEFINE_MAP(state_of_charge,
                voltage,
                energy_consumed,
                flight_time,
                distance_traveled,
                collision_count);

        TripStats()
        {}
        
        TripStats(const msr::airlib::TripStats& s)
        {
            state_of_charge = s.state_of_charge;
            voltage = s.voltage;
            energy_consumed = s.energy_consumed;
            flight_time = s.flight_time;
            distance_traveled = s.distance_traveled;
            collision_count = s.collision_count;
        }
        
        msr::airlib::TripStats to() const
        {
            return msr::airlib::TripStats(state_of_charge,
                    voltage,
                    energy_consumed,
                    flight_time,
                    distance_traveled,
                    collision_count);
       }
    };
    


    struct MultirotorState {
        CollisionInfo collision;
        KinematicsState kinematics_estimated;
        KinematicsState kinematics_true;
        GeoPoint gps_location;
        uint64_t timestamp;
        LandedState landed_state;
        RCData rc_data;
        std::vector<std::string> controller_messages;
        TripStats trip_stats;

        MSGPACK_DEFINE_MAP(collision, kinematics_estimated, gps_location, timestamp, landed_state, rc_data, trip_stats);

        MultirotorState()
        {}

        MultirotorState(const msr::airlib::MultirotorState& s)
        {
            collision = s.collision;
            kinematics_estimated = s.kinematics_estimated;
            gps_location = s.gps_location;
            timestamp = s.timestamp;
            landed_state = s.landed_state;
            rc_data = RCData(s.rc_data);
            trip_stats = s.trip_stats; 
        }

        msr::airlib::MultirotorState to() const
        {
            return msr::airlib::MultirotorState(collision.to(), kinematics_estimated.to(), 
                gps_location.to(), timestamp, landed_state, rc_data.to(), trip_stats.to());
        }
    };

};

}} //namespace

MSGPACK_ADD_ENUM(msr::airlib::DrivetrainType);
MSGPACK_ADD_ENUM(msr::airlib::LandedState);


#endif
