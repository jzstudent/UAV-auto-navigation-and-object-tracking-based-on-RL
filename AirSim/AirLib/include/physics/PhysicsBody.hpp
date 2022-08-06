// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef airsim_core_PhysicsBody_hpp
#define airsim_core_PhysicsBody_hpp

#include "common/Common.hpp"
#include "common/UpdatableObject.hpp"
#include "PhysicsBodyVertex.hpp"
#include "common/CommonStructs.hpp"
#include "Kinematics.hpp"
#include "Environment.hpp"
#include <unordered_set>
#include "Battery.hpp"
#include <cmath>
#include <exception>
#include <cmath>

namespace msr { namespace airlib {

class PhysicsBody : public UpdatableObject {
public: //interface
    virtual real_T getRestitution() const = 0;
    virtual real_T getFriction() const = 0;

    //derived class may return covariant type
    virtual uint wrenchVertexCount() const
    {
        return 0;
    }

    virtual EnergyRotorSpecs getEnergyRotorSpecs(){
        EnergyRotorSpecs energy_rotor_specs;     
        return  energy_rotor_specs;
    } 
    
    virtual PhysicsBodyVertex& getWrenchVertex(uint index)
    {
        unused(index);
        throw std::out_of_range("no physics vertex are available");
    }
    virtual const PhysicsBodyVertex& getWrenchVertex(uint index) const
    {
        unused(index);
        throw std::out_of_range("no physics vertex are available");
    }

    virtual uint dragVertexCount() const
    {
        return 0;
    }
    virtual PhysicsBodyVertex& getDragVertex(uint index)
    {
        unused(index);
        throw std::out_of_range("no physics vertex are available");
    }
    virtual const PhysicsBodyVertex& getDragVertex(uint index) const
    {
        unused(index);
        throw std::out_of_range("no physics vertex are available");
    }
    virtual void setCollisionInfo(const CollisionInfo& collision_info)
    {
        collision_info_ = collision_info;
    }

    virtual void updateTime(TTimeDelta dt) {
        total_time_since_creation_ += dt;
    }
    virtual void updateDistanceTraveled(Pose cur_pose) {
        if (distance_traveled_ == -1) { //first value
            distance_traveled_ = 0;
            last_pose_ = cur_pose;
        }

        float distance_traveled_temp = sqrt(pow((cur_pose.position - last_pose_.position)[0],2) + pow((cur_pose.position - last_pose_.position)[1],2) + pow((cur_pose.position - last_pose_.position)[2],2));

        if (distance_traveled_temp > distance_traveled_quanta_) { //only update if greater than certain threshold cause otherwise the error accumulates
            distance_traveled_ += distance_traveled_temp;
            last_pose_ = cur_pose;
        }
    }

    virtual void updateEnergyConsumed(float inst_energy) {
        energy_consumed_ += inst_energy;
    }

    virtual void updateKinematics(const Kinematics::State& state)
    {
        if (VectorMath::hasNan(state.twist.linear)) {
            //Utils::DebugBreak();
            Utils::log("Linear velocity had NaN!", Utils::kLogLevelError);
        }

        kinematics_->setState(state);
        kinematics_->update();
    }


public: //methods
    //constructors
    PhysicsBody()
    {
        //allow default constructor with later call for initialize
    }
    PhysicsBody(real_T mass, const Matrix3x3r& inertia, Kinematics* kinematics, Environment* environment)
    {
        initialize(mass, inertia, kinematics, environment);
    }
    void initialize(real_T mass, const Matrix3x3r& inertia, Kinematics* kinematics, Environment* environment)
    {
        mass_ = mass;
        mass_inv_ = 1.0f / mass;
        inertia_ = inertia;
        inertia_inv_ = inertia_.inverse();
        environment_ = environment;
        kinematics_ = kinematics;
    }

    bool hasBattery() const { return battery_ != nullptr; }
 
    //enable physics body detection
    virtual UpdatableObject* getPhysicsBody() override
    {
        return this;
    }


    //*** Start: UpdatableState implementation ***//
    virtual void reset() override
    {
        UpdatableObject::reset();

        if (environment_)
            environment_->reset();
        wrench_ = Wrench::zero();
        collision_info_ = CollisionInfo();
        collision_response_ = CollisionResponse();
		grounded_ = false;

        //update individual vertices
        for (uint vertex_index = 0; vertex_index < wrenchVertexCount(); ++vertex_index) {
            getWrenchVertex(vertex_index).reset();
        }
        for (uint vertex_index = 0; vertex_index < dragVertexCount(); ++vertex_index) {
            getDragVertex(vertex_index).reset();
        }
    }

    virtual void update() override
    {
        UpdatableObject::update();

        //update individual vertices - each vertex takes control signal as input and
        //produces force and thrust as output
        for (uint vertex_index = 0; vertex_index < wrenchVertexCount(); ++vertex_index) {
            getWrenchVertex(vertex_index).update();
        }
        for (uint vertex_index = 0; vertex_index < dragVertexCount(); ++vertex_index) {
            getDragVertex(vertex_index).update();
        }
    }

    virtual void reportState(StateReporter& reporter) override
    {
        //call base
        UpdatableObject::reportState(reporter);

        reporter.writeHeading("Kinematics");
    }
    //*** End: UpdatableState implementation ***//


    //getters
    real_T getMass()  const
    {
        return mass_;
    }
    real_T getMassInv()  const
    {
        return mass_inv_;
    }    
    const Matrix3x3r& getInertia()  const
    {
        return inertia_;
    }
    const Matrix3x3r& getInertiaInv()  const
    {
        return inertia_inv_;
    }

    const Pose& getPose() const
    {
        return kinematics_->getPose();
    }
    void setPose(const Pose& pose)
    {
        return kinematics_->setPose(pose);
    }
    const Twist& getTwist() const
    {
        return kinematics_->getTwist();
    }
    void setTwist(const Twist& twist)
    {
        return kinematics_->setTwist(twist);
    }

    powerlib::Battery *getBattery() { return battery_; }
 	
 	float getStateOfCharge() const
    {
        if (battery_ != nullptr) {
            return battery_->StateOfCharge();
        } else {
            return -100.0;
        }
    }
 
    float getVotage() const
    {
        if (battery_ != nullptr) {
            return battery_->Voltage();
        } else {
            return 0.0;
        }
    }
 
    float getNominalVoltage() const
    {
        if (battery_ != nullptr) {
            return battery_->NominalVoltage();
        } else {
            return 0.0;
        }
    }
 
    float getCapacity() const
    {
        if (battery_ != nullptr) {
            return battery_->Capacity();
        } else {
            return 0.0;
        }
    }
 
    float getDistanceTraveled() const
    {
        return distance_traveled_;
    }
 
     
    float getEnergyConsumed() const
    {
        return energy_consumed_;
    }
 
    int getCollisionCount() const                                                                       {                                                                                               return collision_response_.collision_count_non_resting;
    }

    float getTotalTime() const
    {
        return (float) total_time_since_creation_;
    }

    const Kinematics::State& getKinematics() const
    {
        return kinematics_->getState();
    }

    const Kinematics::State& getInitialKinematics() const
    {
        return kinematics_->getInitialState();
    }
    const Environment& getEnvironment() const
    {
        return *environment_;
    }
    Environment& getEnvironment()
    {
        return *environment_;
    }
    bool hasEnvironment() const
    {
        return environment_ != nullptr;
    }
    const Wrench& getWrench() const
    {
        return wrench_;
    }
    void setWrench(const Wrench&  wrench)
    {
        wrench_ = wrench;
    }
    const CollisionInfo& getCollisionInfo() const
    {
        return collision_info_;
    }

    const CollisionResponse& getCollisionResponseInfo() const
    {
        return collision_response_;
    }
    CollisionResponse& getCollisionResponseInfo()
    {
        return collision_response_;
    }

	bool isGrounded() const
	{
		return grounded_;
	}
	void setGrounded(bool grounded)
	{
		grounded_ = grounded;
	}

public:
    //for use in physics engine: //TODO: use getter/setter or friend method?
    TTimePoint last_kinematics_time;

private:
    real_T mass_, mass_inv_;
    Matrix3x3r inertia_, inertia_inv_;
    TTimeDelta total_time_since_creation_ = 0;
    Pose last_pose_; 
    float distance_traveled_ = -1.0f;
    float distance_traveled_quanta_ = .1f; //the smallest amount that ould be accumulated to the distance traveled. This is set to cancel the accumulated error
    float energy_consumed_ = 0;


    Kinematics* kinematics_ = nullptr;
    Environment* environment_ = nullptr;

    //force is in world frame but torque is not
    Wrench wrench_;

    CollisionInfo collision_info_;
    CollisionResponse collision_response_;

	bool grounded_ = false;

protected:
    powerlib::Battery* battery_ = nullptr;
};

}} //namespace
#endif
