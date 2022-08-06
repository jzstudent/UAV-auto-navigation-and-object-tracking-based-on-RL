// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef msr_airlib_multirotor_hpp
#define msr_airlib_multirotor_hpp

#include "common/Common.hpp"
#include "common/CommonStructs.hpp"
#include "Rotor.hpp"
#include "api/VehicleApiBase.hpp"
#include "api/VehicleSimApiBase.hpp"
#include "MultiRotorParams.hpp"
#include <vector>
#include "physics/PhysicsBody.hpp"
#include "common/Settings.hpp"

#ifndef DEFAULT_VOLTAGE
#define DEFAULT_VOLTAGE (11.1f)
#endif  // DEFAULT_VOLTAGE

#ifndef DEFAULT_CAPACITY
#define DEFAULT_CAPACITY (5.5f)
#endif  // DEFAULT_CAPACITY
 

namespace msr { namespace airlib {

class MultiRotor : public PhysicsBody {
public:
    MultiRotor(MultiRotorParams* params, MultirotorApiBase* vehicle_api, 
        Kinematics* kinematics, Environment* environment)
        : params_(params), vehicle_api_(vehicle_api)
    {
        initialize(kinematics, environment);
    }

    //*** Start: UpdatableState implementation ***//
    virtual void reset() override
    {
        //reset rotors, kinematics and environment
        PhysicsBody::reset();

        //reset sensors last after their ground truth has been reset
        resetSensors();
        // reset battery
        if (battery_ != nullptr) {
            battery_->reset();
        }
    }

    void setEnergyRotorSpecs(Settings& settings){ 

        Settings energy_model_settings;
        //if (!settings.getChild("EnergyModelSettings", energy_model_settings)) {
        //UAirBlueprintLib::LogMessage(TEXT("Energy model settings not provided. Energy values must  be ignored"), "", LogDebugLevel::Failure);
        //}
        settings.getChild("EnergyModelSettings", energy_model_settings);

        energy_rotor_specs_.set_mass(float(energy_model_settings.getFloat("mass", 0)));
        energy_rotor_specs_.set_mass_coeff(float(energy_model_settings.getFloat("mass_coeff", 0)));
        energy_rotor_specs_.set_vxy_coeff(float(energy_model_settings.getFloat("vxy_coeff", 0)));
        energy_rotor_specs_.set_axy_coeff(float(energy_model_settings.getFloat("axy_coeff", 0)));
        energy_rotor_specs_.set_vxy_axy_coeff(float(energy_model_settings.getFloat("vxy_axy_coeff", 0)));
        energy_rotor_specs_.set_vz_coeff(float(energy_model_settings.getFloat("vz_coeff", 0)));
        energy_rotor_specs_.set_az_coeff(float(energy_model_settings.getFloat("az_coeff", 0)));
        energy_rotor_specs_.set_vz_az_coeff(float(energy_model_settings.getFloat("vz_az_coeff", 0)));
        energy_rotor_specs_.set_one_coeff(float(energy_model_settings.getFloat("one_coeff", 0)));
        energy_rotor_specs_.set_vxy_wxy_coeff(float(energy_model_settings.getFloat("vxy_wxy_coeff", 0)));

    }

    EnergyRotorSpecs getEnergyRotorSpecs(){
        return energy_rotor_specs_;
     }
 
    virtual void update() override
    {
        //update forces on vertices that we will use next
        PhysicsBody::update();

        //Note that controller gets updated after kinematics gets updated in updateKinematics
        //otherwise sensors will have values from previous cycle causing lags which will appear
        //as crazy jerks whenever commands like velocity is issued
    }
    virtual void reportState(StateReporter& reporter) override
    {
        //call base
        PhysicsBody::reportState(reporter);

        reportSensors(*params_, reporter);

        //report rotors
        for (uint rotor_index = 0; rotor_index < rotors_.size(); ++rotor_index) {
            reporter.startHeading("", 1);
            reporter.writeValue("Rotor", rotor_index);
            reporter.endHeading(false, 1);
            rotors_.at(rotor_index).reportState(reporter);
        }
    }
    //*** End: UpdatableState implementation ***//


    //Physics engine calls this method to set next kinematics
    virtual void updateKinematics(const Kinematics::State& kinematics) override
    {
        PhysicsBody::updateKinematics(kinematics);

        updateSensors(*params_, getKinematics(), getEnvironment());

        //update controller which will update actuator control signal
        vehicle_api_->update();

        //transfer new input values from controller to rotors
        for (uint rotor_index = 0; rotor_index < rotors_.size(); ++rotor_index) {
            rotors_.at(rotor_index).setControlSignal(
                vehicle_api_->getActuation(rotor_index));
        }
    
        if (battery_ != nullptr) {
            TripStats trip_stats; 
            trip_stats.state_of_charge = battery_->StateOfCharge();
            trip_stats.voltage = battery_->Voltage();
            trip_stats.energy_consumed = getEnergyConsumed();
            trip_stats.collision_count = getCollisionCount();
            trip_stats.flight_time = getTotalTime();
            trip_stats.distance_traveled = getDistanceTraveled();
            vehicle_api_->setTripStats(trip_stats);
        }
    }

    //sensor getter
    const SensorCollection& getSensors() const
    {
        return params_->getSensors();
    }

    //physics body interface
    virtual uint wrenchVertexCount() const  override
    {
        return params_->getParams().rotor_count;
    }
    virtual PhysicsBodyVertex& getWrenchVertex(uint index)  override
    {
        return rotors_.at(index);
    }
    virtual const PhysicsBodyVertex& getWrenchVertex(uint index) const override
    {
        return rotors_.at(index);
    }

    virtual uint dragVertexCount() const override
    {
        return static_cast<uint>(drag_vertices_.size());
    }
    virtual PhysicsBodyVertex& getDragVertex(uint index)  override
    {
        return drag_vertices_.at(index);
    }
    virtual const PhysicsBodyVertex& getDragVertex(uint index) const override
    {
        return drag_vertices_.at(index);
    }

    virtual real_T getRestitution() const override
    {
        return params_->getParams().restitution;
    }
    virtual real_T getFriction()  const override
    {
        return params_->getParams().friction;
    }

    Rotor::Output getRotorOutput(uint rotor_index) const
    {
        return rotors_.at(rotor_index).getOutput();
    }

    virtual ~MultiRotor() = default;

private: //methods
    void initialize(Kinematics* kinematics, Environment* environment)
    {
        PhysicsBody::initialize(params_->getParams().mass, params_->getParams().inertia, kinematics, environment);

        Settings& settings = Settings::singleton();
        setEnergyRotorSpecs(settings);  //set energy coeffs
        float v = float(settings.getFloat("BatteryVoltage", DEFAULT_VOLTAGE));
        float c = float(settings.getFloat("BatteryCapacity", DEFAULT_CAPACITY));
        battery_ = new powerlib::Battery(v, c);
 
        
        createRotors(*params_, rotors_, environment);
        createDragVertices();

        initSensors(*params_, getKinematics(), getEnvironment());
    }

    static void createRotors(const MultiRotorParams& params, vector<Rotor>& rotors, const Environment* environment)
    {
        rotors.clear();
        //for each rotor pose
        for (uint rotor_index = 0; rotor_index < params.getParams().rotor_poses.size(); ++rotor_index) {
            const MultiRotorParams::RotorPose& rotor_pose = params.getParams().rotor_poses.at(rotor_index);
            rotors.emplace_back(rotor_pose.position, rotor_pose.normal, rotor_pose.direction, params.getParams().rotor_params, environment, rotor_index);
        }
    }

    void reportSensors(MultiRotorParams& params, StateReporter& reporter)
    {
        params.getSensors().reportState(reporter);
    }

    void updateSensors(MultiRotorParams& params, const Kinematics::State& state, const Environment& environment)
    {
        unused(state);
        unused(environment);
        params.getSensors().update();
    }

    void initSensors(MultiRotorParams& params, const Kinematics::State& state, const Environment& environment)
    {
        params.getSensors().initialize(&state, &environment);
    }

    void resetSensors()
    {
        params_->getSensors().reset();
    }

    /* 
    TripStats getTripStats(){
        return trip_stats_;
    }
    */
   
    void createDragVertices()
    {
        const auto& params = params_->getParams();

        //Drone is seen as central body that is connected to propellers via arm. We approximate central body as box of size x, y, z.
        //The drag depends on area exposed so we also add area of propellers to approximate drag they may introduce due to their area.
        //while moving along any axis, we find area that will be exposed in that direction
        real_T propeller_area = M_PIf * params.rotor_params.propeller_diameter * params.rotor_params.propeller_diameter;
        real_T propeller_xsection = M_PIf * params.rotor_params.propeller_diameter * params.rotor_params.propeller_height;

        real_T top_bottom_area = params.body_box.x() * params.body_box.y();
        real_T left_right_area = params.body_box.x() * params.body_box.z();
        real_T front_back_area = params.body_box.y() * params.body_box.z();
        Vector3r drag_factor_unit = Vector3r(
            front_back_area + rotors_.size() * propeller_xsection, 
            left_right_area + rotors_.size() * propeller_xsection, 
            top_bottom_area + rotors_.size() * propeller_area) 
            * params.linear_drag_coefficient / 2; 

        //add six drag vertices representing 6 sides
        drag_vertices_.clear();
        drag_vertices_.emplace_back(Vector3r(0, 0, -params.body_box.z()), Vector3r(0, 0, -1), drag_factor_unit.z());
        drag_vertices_.emplace_back(Vector3r(0, 0,  params.body_box.z()), Vector3r(0, 0,  1), drag_factor_unit.z());
        drag_vertices_.emplace_back(Vector3r(0, -params.body_box.y(), 0), Vector3r(0, -1, 0), drag_factor_unit.y());
        drag_vertices_.emplace_back(Vector3r(0,  params.body_box.y(), 0), Vector3r(0,  1, 0), drag_factor_unit.y());
        drag_vertices_.emplace_back(Vector3r(-params.body_box.x(), 0, 0), Vector3r(-1, 0, 0), drag_factor_unit.x());
        drag_vertices_.emplace_back(Vector3r( params.body_box.x(), 0, 0), Vector3r( 1, 0, 0), drag_factor_unit.x());

    }

private: //fields
    MultiRotorParams* params_;

    //let us be the owner of rotors object
    vector<Rotor> rotors_;
    vector<PhysicsBodyVertex> drag_vertices_;
    EnergyRotorSpecs energy_rotor_specs_;
    std::unique_ptr<Environment> environment_;
    MultirotorApiBase* vehicle_api_;
 //    TripStats trip_stats_;
};

}} //namespace
#endif
