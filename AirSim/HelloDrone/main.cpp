// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "common/common_utils/StrictMode.hpp"
STRICT_MODE_OFF
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif // !RPCLIB_MSGPACK
#include "rpc/rpc_error.h"
STRICT_MODE_ON

#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include "common/common_utils/FileSystem.hpp"
#include <iostream>
#include <chrono>

#include <regex>

int main(int argc, char **argv)
{

    // read path from file
    if(argc != 2)
    {
        std::cerr << std::endl << "Usage: ./HelloDrone path_to( airsim_record.txt)" << std::endl;
        return -1;
    }

    std::string airsim_file_path = std::string((argv[1]));
    std::cerr << "airsim_file_path:" << airsim_file_path <<std::endl;
    std::ifstream in_file;
    in_file.open(airsim_file_path.data());
    assert(in_file.is_open());

    std::vector<Vector3r> read_path;

    std::string current_line;
    int line_count = 0;
    Eigen::Quaterniond last_q;
    while(std::getline(in_file, current_line)) {

        std::regex tab_re("\t");
        std::vector<std::string> v(std::sregex_token_iterator(current_line.begin(), current_line.end(), tab_re, -1),
                                   std::sregex_token_iterator());

        line_count++;
        if(line_count == 1) {
            last_q = Eigen::Quaterniond(std::atof(v[4].c_str()),
                                        std::atof(v[5].c_str()), std::atof(v[6].c_str()), std::atof(v[7].c_str()));
        }
        //if(line_count % 5 != 0)
            //continue;

        Eigen::Quaterniond q = Eigen::Quaterniond(std::atof(v[4].c_str()),
                                                  std::atof(v[5].c_str()), std::atof(v[6].c_str()), std::atof(v[7].c_str()));
        Eigen::Vector3d error = 2 * (last_q.inverse() * q).vec();

        std::cout << "last_q: " << std::endl << last_q.coeffs().transpose() << std::endl;
        std::cout << "q: " << std::endl << q.coeffs().transpose() << std::endl;
        std::cout << "error: " << std::endl << error.transpose() << ",  error normal:" << error.norm() << std::endl << std::endl;

        if(error.norm() > 0.05 ) {
            float x = std::atof(v[1].c_str());
            float y = std::atof(v[2].c_str());
            float z = std::atof(v[3].c_str());
            read_path.emplace_back(Vector3r{x, y, z});
            last_q = q;
        }

    }


    using namespace msr::airlib;

    msr::airlib::MultirotorRpcLibClient client;
    typedef ImageCaptureBase::ImageRequest ImageRequest;
    typedef ImageCaptureBase::ImageResponse ImageResponse;
    typedef ImageCaptureBase::ImageType ImageType;
    typedef common_utils::FileSystem FileSystem;
    
    try {
        client.confirmConnection();

        std::cout << "Press Enter to get FPV image" << std::endl; std::cin.get();
        vector<ImageRequest> request = { ImageRequest("0", ImageType::Scene), ImageRequest("1", ImageType::DepthPlanner, true) };
        const vector<ImageResponse>& response = client.simGetImages(request);
        std::cout << "# of images received: " << response.size() << std::endl;

        if (response.size() > 0) {
            std::cout << "Enter path with ending separator to save images (leave empty for no save)" << std::endl; 
            std::string path;
            std::getline(std::cin, path);

            for (const ImageResponse& image_info : response) {
                std::cout << "Image uint8 size: " << image_info.image_data_uint8.size() << std::endl;
                std::cout << "Image float size: " << image_info.image_data_float.size() << std::endl;

                if (path != "") {
                    std::string file_path = FileSystem::combine(path, std::to_string(image_info.time_stamp));
                    if (image_info.pixels_as_float) {
                        Utils::writePfmFile(image_info.image_data_float.data(), image_info.width, image_info.height,
                            file_path + ".pfm");
                    }
                    else {
                        std::ofstream file(file_path + ".png", std::ios::binary);
                        file.write(reinterpret_cast<const char*>(image_info.image_data_uint8.data()), image_info.image_data_uint8.size());
                        file.close();
                    }
                }
            }
        }

        std::cout << "Press Enter to arm the drone" << std::endl; std::cin.get();
        client.enableApiControl(true);
        client.armDisarm(true);

        /*auto barometer_data = client.getBarometerData();
        std::cout << "Barometer data \n" 
            << "barometer_data.time_stamp \t" << barometer_data.time_stamp << std::endl
            << "barometer_data.altitude \t" << barometer_data.altitude << std::endl
            << "barometer_data.pressure \t" << barometer_data.pressure << std::endl
            << "barometer_data.qnh \t" << barometer_data.qnh << std::endl;
        auto imu_data = client.getImuData();
        std::cout << "IMU data \n"
            << "imu_data.time_stamp \t" << imu_data.time_stamp << std::endl
            << "imu_data.orientation \t" << imu_data.orientation << std::endl
            << "imu_data.angular_velocity \t" << imu_data.angular_velocity << std::endl
            << "imu_data.linear_acceleration \t" << imu_data.linear_acceleration << std::endl;

        auto gps_data = client.getGpsData();
        std::cout << "GPS data \n"
            << "gps_data.time_stamp \t" << gps_data.time_stamp << std::endl
            << "gps_data.gnss.time_utc \t" << gps_data.gnss.time_utc << std::endl
            << "gps_data.gnss.geo_point \t" << gps_data.gnss.geo_point << std::endl
            << "gps_data.gnss.eph \t" << gps_data.gnss.eph << std::endl
            << "gps_data.gnss.epv \t" << gps_data.gnss.epv << std::endl
            << "gps_data.gnss.velocity \t" << gps_data.gnss.velocity << std::endl
            << "gps_data.gnss.fix_type \t" << gps_data.gnss.fix_type << std::endl;

        auto magnetometer_data = client.getMagnetometerData();
        std::cout << "Magnetometer data \n"
            << "magnetometer_data.time_stamp \t" << magnetometer_data.time_stamp << std::endl
            << "magnetometer_data.magnetic_field_body \t" << magnetometer_data.magnetic_field_body << std::endl;
            // << "magnetometer_data.magnetic_field_covariance" << magnetometer_data.magnetic_field_covariance // not implemented in sensor*/


        /*double period = 0.01;
        char ccc;
        do {
            //msr::airlib::MultirotorRpcLibClient client;  // or msr::airlib::CarRpcLibClient
            msr::airlib::Kinematics::State k_state = client.simGetGroundTruthKinematics();
            msr::airlib::Environment::State e_state = client.simGetGroundTruthEnvironment();

            // timestamp
            auto t = Utils::getTimeSinceEpochSecs();

            // angular velocity of IMU - angular velocity in the body frame
            msr::airlib::Vector3r w_b = k_state.twist.angular;
            // linear acceleration of body in world frame
            msr::airlib::Vector3r la_b_w =  k_state.accelerations.linear;
            // subtract gravity
            la_b_w = la_b_w - e_state.gravity;
            // get acceleration in the body frame - IMU linear acceleration
            msr::airlib::Vector3r la_b = msr::airlib::VectorMath::transformToBodyFrame(la_b_w, k_state.pose.orientation, true);

            // add noise and bias as shown in the above example file
            std::cout << std::fixed << std::setprecision(10);
            std::cout << "IMU data \n"
                      << "imu_data.time_stamp \t" << t << std::endl
                      << "imu_data.angular_velocity \t" << w_b << std::endl
                      << "imu_data.linear_acceleration \t" << la_b << std::endl;

            // add sleep here to control the frame rate if necessary
            std::this_thread::sleep_for(std::chrono::duration<double>(period));
            ccc = getchar();

        }while (ccc != 'c');*/


        std::cout << "Press Enter to takeoff" << std::endl; std::cin.get();
        float takeoffTimeout = 5; 
        client.takeoffAsync(takeoffTimeout)->waitOnLastTask();

        // switch to explicit hover mode so that this is the fall back when 
        // move* commands are finished.
        std::this_thread::sleep_for(std::chrono::duration<double>(5));
        client.hoverAsync()->waitOnLastTask();

        //std::cout << "Press Enter to fly in a 10m box pattern at 3 m/s velocity" << std::endl; std::cin.get();
        // moveByVelocityZ is an offboard operation, so we need to set offboard mode.
        client.enableApiControl(true);
        DrivetrainType driveTrain = DrivetrainType::MaxDegreeOfFreedom;
        YawMode yaw_mode(false, 0);

        auto position = client.getMultirotorState().getPosition();
        float x = position.x();
        float y = position.y();
        float z = position.z(); // current position (NED coordinate system).  
        /*const float speed = 3.0f;
        const float size = 10.0f; 
        const float duration = size / speed;

        std::cout << "moveByVelocityZ(" << speed << ", 0, " << z << "," << duration << ")" << std::endl;
        client.moveByVelocityZAsync(speed, 0, z, duration, driveTrain, yaw_mode);
        std::this_thread::sleep_for(std::chrono::duration<double>(duration));
        std::cout << "moveByVelocityZ(0, " << speed << "," << z << "," << duration << ")" << std::endl;
        client.moveByVelocityZAsync(0, speed, z, duration, driveTrain, yaw_mode);
        std::this_thread::sleep_for(std::chrono::duration<double>(duration));
        std::cout << "moveByVelocityZ(" << -speed << ", 0, " << z << "," << duration << ")" << std::endl;
        client.moveByVelocityZAsync(-speed, 0, z, duration, driveTrain, yaw_mode);
        std::this_thread::sleep_for(std::chrono::duration<double>(duration));
        std::cout << "moveByVelocityZ(0, " << -speed << "," << z << "," << duration << ")" << std::endl;
        client.moveByVelocityZAsync(0, -speed, z, duration, driveTrain, yaw_mode);
        std::this_thread::sleep_for(std::chrono::duration<double>(duration));*/

        std::cout << "Press Enter to moveToPosition" << std::endl; std::cin.get();

        //std::vector<Vector3r> path = {Vector3r(20, 0, z-5), Vector3r(20, -20, z-5), Vector3r(0, -20, z-5), Vector3r(0, 0, z-5)};

        /*for(auto& position : read_path) {
            std::cout << "moveToPosition(" << position.x() << ", " << position.y() << ", " << position.z()
                        << ", velocity: " << 2 << ", max duration: " << 15 << ")" << std::endl;
            client.moveToPositionAsync(position.x(), position.y(), position.z(), 2, 15, driveTrain, yaw_mode, -1)->waitOnLastTask();
            //std::cout << "test" << std::endl;
            msr::airlib::Kinematics::State k_state = client.simGetGroundTruthKinematics();
            msr::airlib::Environment::State e_state = client.simGetGroundTruthEnvironment();
            Vector3r k_state_position = k_state.pose.position;
            Vector3r e_state_position = e_state.position;
            std::cout << "Pose \n"
                      << "k_state_position \t" << k_state_position << std::endl
                      << "e_state_position \t" << e_state_position << std::endl;
        }*/

        /*DrivetrainType driveTrain2 = DrivetrainType::MaxDegreeOfFreedom;
        YawMode yaw_mode2(false, 0);

        client.moveByRollPitchYawZAsync(0.3, 0, 0, z, 3)->waitOnLastTask();
        //client.moveByRollPitchYawThrottleAsync(0.5, 0, 0, 0.1, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();
        client.moveByRollPitchYawZAsync(0, 0, 0, z, 3)->waitOnLastTask();
        //client.moveByRollPitchYawThrottleAsync(0, 0, 0, 0.1, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();

        client.moveByRollPitchYawZAsync(-0.3, 0, 0, z, 3)->waitOnLastTask();
        //client.moveByRollPitchYawThrottleAsync(-0.5, 0, 0, 0.1, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();
        client.moveByRollPitchYawZAsync(0, 0, 0, z, 3)->waitOnLastTask();
        //client.moveByRollPitchYawThrottleAsync(0, 0, 0, 0.1, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();

        client.moveByRollPitchYawZAsync(0, 0.2, 0, z, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();
        client.moveByRollPitchYawZAsync(0, 0, 0, z, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();

        client.moveByRollPitchYawZAsync(0, -0.2, 0, z, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();
        client.moveByRollPitchYawZAsync(0, 0, 0, z, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();

        client.moveByRollPitchYawZAsync(0, 0, 0.3, z, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();
        client.moveByRollPitchYawZAsync(0, 0, 0, z, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();

        client.moveByRollPitchYawZAsync(0, 0, -0.3
                                        , z, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();
        client.moveByRollPitchYawZAsync(0, 0, 0, z, 3)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.1, 5, driveTrain2, yaw_mode2)->waitOnLastTask();*/

        /*client.moveByAngleRatesZAsync(0.01, 0, 0, z, 8)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.01)->waitOnLastTask();
        client.moveByAngleRatesZAsync(-0.01, 0, 0, z, 8)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.01)->waitOnLastTask();
        client.moveByAngleRatesZAsync(0, 0.01, 0, z, 8)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.01)->waitOnLastTask();
        client.moveByAngleRatesZAsync(0, -0.01, 0, z, 8)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.01)->waitOnLastTask();
        client.moveByAngleRatesZAsync(0, 0, 0.01, z, 8)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.01)->waitOnLastTask();
        client.moveByAngleRatesZAsync(0, 0, -0.01, z, 8)->waitOnLastTask();
        client.moveToPositionAsync(x,y,z,0.01)->waitOnLastTask();*/

        auto iter = read_path.begin();
        auto next_iter = std::next(iter);
        for(; next_iter != read_path.end(); iter++, next_iter++ ) {

            std::vector<Vector3r> seg_path;
            seg_path.push_back(*iter);
            seg_path.push_back(*next_iter);
            std::cout << "moveOnPath( position1: " << iter->x() << ", " << iter->y() << ", " << iter->z()
                      << " position2: " << next_iter->x() << ", " << next_iter->y() << ", " << next_iter->z()
                      << ", velocity: " << 3 << ", max duration: " << 50 << ")" << std::endl;
            client.moveOnPathAsync(seg_path, 3, 70, driveTrain, yaw_mode, 10)->waitOnLastTask();
            //std::cout << "test" << std::endl;
            msr::airlib::Kinematics::State k_state = client.simGetGroundTruthKinematics();
            msr::airlib::Environment::State e_state = client.simGetGroundTruthEnvironment();
            Vector3r k_state_position = k_state.pose.position;
            Vector3r e_state_position = e_state.position;
            std::cout << "Pose \n"
                      << "k_state_position \t" << k_state_position << std::endl
                      << "e_state_position \t" << e_state_position << std::endl;
        }

        //std::cout << "Press Enter to moveOnPath" << std::endl; std::cin.get();
        //const float my_speed = 1.0f;
        //std::vector my_path = { Vector3r(20, 0, z-5), Vector3r(20, -20, z-5)};
        //std::vector my_path_2 = { Vector3r(0, -20, z-5), Vector3r(0, 0, z-5)};
        //std::vector my_path_3 = {  Vector3r(20, 20, z-5), Vector3r(0, 20, z-5)};
        //client.moveOnPathAsync(my_path, my_speed, 70, driveTrain, yaw_mode, 2.0)->waitOnLastTask();
        //client.moveOnPathAsync(my_path_2, my_speed, 70, driveTrain, yaw_mode, 2.0)->waitOnLastTask();
        //std::cout << "moveToPosition" << std::endl;
        //client.moveToPositionAsync(10, 10, -10, 1)->waitOnLastTask();
        // std::this_thread::sleep_for(std::chrono::duration<double>(50));
        // test get IMU during moveOnPath

        client.hoverAsync()->waitOnLastTask();

        std::cout << "Press Enter to land" << std::endl; std::cin.get();
        client.landAsync()->waitOnLastTask();

        std::cout << "Press Enter to disarm" << std::endl; std::cin.get();
        client.armDisarm(false);

    }
    catch (rpc::rpc_error&  e) {
        std::string msg = e.get_error().as<std::string>();
        std::cout << "Exception raised by the API, something went wrong." << std::endl << msg << std::endl;
    }

    return 0;
}
