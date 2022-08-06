// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef msr_airlib_vehicles_SimpleFlightQuadX_hpp
#define msr_airlib_vehicles_SimpleFlightQuadX_hpp

#include "vehicles/multirotor/firmwares/simple_flight/SimpleFlightApi.hpp"
#include "vehicles/multirotor/MultiRotorParams.hpp"
#include "common/AirSimSettings.hpp"
#include "sensors/SensorFactory.hpp"
#define MATRICE_100

namespace msr { namespace airlib {

class SimpleFlightQuadXParams : public MultiRotorParams {
public:
    SimpleFlightQuadXParams(const AirSimSettings::VehicleSetting* vehicle_setting, std::shared_ptr<const SensorFactory> sensor_factory)
        : vehicle_setting_(vehicle_setting), sensor_factory_(sensor_factory)
    {
    }

    virtual ~SimpleFlightQuadXParams() = default;

    virtual std::unique_ptr<MultirotorApiBase> createMultirotorApi() override
    {
        return std::unique_ptr<MultirotorApiBase>(new SimpleFlightApi(this, vehicle_setting_));
    }

protected:
    virtual void setupParams() override
    {
        auto& params = getParams();

        /******* Below is same config as PX4 generic model ********/
#ifdef F450
        /* TODO: calculate and un-comment max_rpm and propeller_diameter */
        params.rotor_count = 4;
        params.mass = 1.0f;
        real_T motor_assembly_weight = 0.055f;
        real_T motor_arm_length = 0.2275f;
        params.rotor_params.C_P = 0.040164f; // the torque co-efficient
        params.rotor_params.C_T = 0.109919f; // the thrust co-efficient
        // params.rotor_params.max_rpm = 13024 * .6; // RPM can be obtained by KV * Voltage
        // params.rotor_params.propeller_diameter = 0.254f; // in meters

        params.body_box.x() = 0.180f; params.body_box.y() = 0.11f; params.body_box.z() = 0.040f;
        real_T rotor_z = 2.5f / 100;
#elif defined(_3DR_SOLO)
        /* TODO: make a new rotor_params and update it with the correct coefficents, rpm, and prop diameter */
        params.rotor_count = 4;
        params.mass = 1.5f;
        real_T motor_assembly_weight = 0.0680389f;
        real_T motor_arm_length = .130175f;
        params.rotor_params.C_P = 0.040164f; // the torque co-efficient
        params.rotor_params.C_T = 0.109919f; // the thrust co-efficient
        params.rotor_params.max_rpm = 13024 * .6; //RPM can be obtained by KV * Voltage
        params.rotor_params.propeller_diameter = 0.254f; //in meters

        params.body_box.x() = .2428875f; params.body_box.y() = .10795; params.body_box.z() = .066675;
        real_T rotor_z = 0.9525f / 100;
#elif defined(MATRICE_100)
        /* TODO: recalculate thrust coefficent to account for propeller efficency */
        params.rotor_count = 4;
        params.mass = 2.431f;
        real_T motor_assembly_weight = 0.106f;
        real_T motor_arm_length = .325f;
        params.rotor_params.C_P = .033526f; // the torque co-efficient
        params.rotor_params.C_T = .027122f; // the thrust co-efficient
        params.rotor_params.max_rpm = 8750; // RPM can be obtained by KV * Voltage (or from docs)
        params.rotor_params.propeller_diameter = 0.3302f; //in meters

        params.body_box.x() = .252f; params.body_box.y() = .190f; params.body_box.z() = .131f;
        real_T rotor_z = .0569f;
#endif
 
         //set up arm lengths
         //dimensions are for F450 frame: http://artofcircuits.com/product/quadcopter-frame-hj450-with-power-distribution
        std::vector<real_T> arm_lengths(params.rotor_count, motor_arm_length);
 
         //set up mass

        //set up arm lengths
        //dimensions are for F450 frame: http://artofcircuits.com/product/quadcopter-frame-hj450-with-power-distribution

        //set up mass
        real_T box_mass = params.mass - params.rotor_count * motor_assembly_weight;

        // using rotor_param default, but if you want to change any of the rotor_params, call calculateMaxThrust() to recompute the max_thrust
        // given new thrust coefficients, motor max_rpm and propeller diameter.
        params.rotor_params.calculateMaxThrust();

        //set up dimensions of core body box or abdomen (not including arms).

        //computer rotor poses
        initializeRotorQuadX(params.rotor_poses, params.rotor_count, arm_lengths.data(), rotor_z);

        //compute inertia matrix
        computeInertiaMatrix(params.inertia, params.body_box, params.rotor_poses, box_mass, motor_assembly_weight);

        //leave everything else to defaults
    }

    virtual const SensorFactory* getSensorFactory() const override
    {
        return sensor_factory_.get();
    }

private:
    vector<unique_ptr<SensorBase>> sensor_storage_;
    const AirSimSettings::VehicleSetting* vehicle_setting_; //store as pointer because of derived classes
    std::shared_ptr<const SensorFactory> sensor_factory_;
};

}} //namespace
#endif
