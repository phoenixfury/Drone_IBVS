#include <assert.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <armadillo>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Twist.h>

#include "fiducial_msgs/Fiducial.h"
#include "fiducial_msgs/FiducialArray.h"
#include "fiducial_msgs/FiducialTransform.h"
#include "fiducial_msgs/FiducialTransformArray.h"

#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/PositionTarget.h>

#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static const std::string OPENCV_WINDOW = "Image window";


class IBVS {        // The class
  private:
    //Aruco marker variables
    fiducial_msgs::FiducialArray fva; // Array of aruco markers
    fiducial_msgs::Fiducial fid_val;    //An aruco marker
    fiducial_msgs::Fiducial * fid = &fid_val;    //An aruco marker
    fiducial_msgs::FiducialTransformArray fta; //Array of aruco marker transforms
    fiducial_msgs::FiducialTransform ft_val; //An aruco marker transform
    fiducial_msgs::FiducialTransform * ft = &ft_val; //An aruco marker transform

    geometry_msgs::Twist ctrl_input; //A twist message containing the control input
    geometry_msgs::PoseStamped pose;
    geometry_msgs::TwistStamped vel_cnt;

    ros::NodeHandle n; //ros node handler

    ros::Publisher pub;
    ros::Subscriber vertices_sub;
    ros::Subscriber transforms_sub;
    ros::Subscriber camera_info_sub;

    // The Drone part
    // Messages
    mavros_msgs::State current_state;
    geometry_msgs::PoseStamped current_pose;
    //Publishers
    ros::Publisher drone_pos_pub;
    ros::Publisher drone_vel_pub;
    //Subscribers
    ros::Subscriber drone_state_sub;
    ros::Subscriber drone_pose_sub;
    //Services
    ros::ServiceClient arming_client;
    ros::ServiceClient set_mode_client;

    //Variables
    mavros_msgs::SetMode offb_set_mode;
    mavros_msgs::CommandBool arm_cmd;
    mavros_msgs::PositionTarget vel_cnt_full;

    // Visualization
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

    // Need optimization
    float w = 1.5;
    arma::mat p_desired = {{480/w, 128/w}, {480/w, 384/w}, {160/w, 384/w}, {160/w, 128/w}};

    // Camera intrinsics
    arma::mat K = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Initialization flags
    bool k_flag = false;
    bool c_flag = false;
    bool m_flag = false;
    bool takeoff_time_init = false;
    bool time_init = false;


    //Loop time variables
    double begin = ros::Time::now().toSec();
    double delta;
    ros::Time last_request = ros::Time::now();

  public:              // Access specifier
    bool takeoff = false;
    IBVS():it_(n)
    {        image_sub_ = it_.subscribe("/fiducial_images", 1, &IBVS::imageCb, this);
             image_pub_ = it_.advertise("/sub_ibvs/output_video", 1);
             cv::namedWindow(OPENCV_WINDOW);
    }

    ~IBVS()
    {cv::destroyWindow(OPENCV_WINDOW);}

    IBVS(ros::NodeHandle *nh):it_(n)
    {
        n = *nh;
        pub = nh->advertise<geometry_msgs::Twist>("jackal_velocity_controller/cmd_vel", 1000);    
        vertices_sub = nh->subscribe("fiducial_vertices", 1000, &IBVS::markerCallback, this);
        transforms_sub = nh->subscribe("fiducial_transforms", 1000, &IBVS::transformCallback, this);
        camera_info_sub = nh->subscribe("/camera/rgb/camera_info", 1, &IBVS::c_infoCallback, this);
        drone_pos_pub =nh->advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local", 10);
        drone_vel_pub =nh->advertise<mavros_msgs::PositionTarget>("mavros/setpoint_raw/local", 10);
        drone_state_sub = nh->subscribe<mavros_msgs::State>("mavros/state", 10, &IBVS::drone_state_callback, this);
        drone_pose_sub =nh->subscribe<geometry_msgs::PoseStamped>("mavros/local_position/pose", 100, &IBVS::drone_pose_callback, this);
        arming_client = nh->serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
        set_mode_client = nh->serviceClient<mavros_msgs::SetMode>("mavros/set_mode");
        // Subscrive to input video feed and publish output video feed
        image_sub_ = it_.subscribe("/fiducial_images", 1, &IBVS::imageCb, this);
        image_pub_ = it_.advertise("/sub_ibvs/output_video", 1);

        cv::namedWindow(OPENCV_WINDOW);
    }
    void c_infoCallback(const sensor_msgs::CameraInfoConstPtr& cam_info)
    {
        if (k_flag == false)
        {
            for (size_t i=0; i<9; i++) 
            {
                K(i) = cam_info->K[i];
            }
            k_flag = true;
        }
    }
    void imageCb(const sensor_msgs::ImageConstPtr& msg)//sensor_msgs::CompressedImage::ConstPtr & msg

      {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
          cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
          ROS_ERROR("cv_bridge exception: %s", e.what());
          return;
        }
//        std::cout<<"hi"<<std::endl;
        // Draw an example circle on the video stream
        if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
          cv::circle(cv_ptr->image, cv::Point(p_desired(0,0),p_desired(0,1)), 3, CV_RGB(0,0,255), -1);
          cv::circle(cv_ptr->image, cv::Point(p_desired(1,0),p_desired(1,1)), 3, CV_RGB(255,128,0), -1);
          cv::circle(cv_ptr->image, cv::Point(p_desired(2,0),p_desired(2,1)), 3, CV_RGB(255,128,0), -1);
          cv::circle(cv_ptr->image, cv::Point(p_desired(3,0),p_desired(3,1)), 3, CV_RGB(255,128,0), -1);

        // Update GUI Window
        cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        cv::waitKey(3);

        // Output modified video stream
        image_pub_.publish(cv_ptr->toImageMsg());
      }


    void drone_state_callback(const mavros_msgs::State::ConstPtr& msg)
    {
        current_state = *msg;
    }


    void drone_pose_callback(const geometry_msgs::PoseStamped::ConstPtr& pose)
    {
        current_pose = *pose;
    }


    void markerCallback(const fiducial_msgs::FiducialArray fva)
    {
        for (size_t i=0; i<fva.fiducials.size(); i++) {
            // fiducial_msgs::Fiducial fid;
            this->fid->fiducial_id = fva.fiducials[i].fiducial_id;

            this->fid->x0 = fva.fiducials[i].x0;
            this->fid->y0 = fva.fiducials[i].y0;
            this->fid->x1 = fva.fiducials[i].x1;
            this->fid->y1 = fva.fiducials[i].y1;
            this->fid->x2 = fva.fiducials[i].x2;
            this->fid->y2 = fva.fiducials[i].y2;
            this->fid->x3 = fva.fiducials[i].x3;
            this->fid->y3 = fva.fiducials[i].y3;
        }
        m_flag = true;
    }


    void transformCallback(const fiducial_msgs::FiducialTransformArray fta)
    {
        for (size_t i=0; i<fta.transforms.size(); i++) {
            // fiducial_msgs::FiducialTransform ft;
            this->ft->fiducial_id = fta.transforms[i].fiducial_id;

            this->ft->transform = fta.transforms[i].transform;

            // ROS_INFO("I got Z: [%f]", this->ft->transform.translation.z);

        }
        c_flag = true;
    }
    bool drone_takeOff()
    {
        geometry_msgs::PoseStamped pose;

        pose.pose.position.x = 0;
        pose.pose.position.y = 0;
        pose.pose.position.z = 1.6;

        if (!takeoff_time_init)
        {
            for(int i = 100; ros::ok() && i > 0; --i)
            {
                delta = ros::Time::now().toSec() - begin;
                if (!takeoff_time_init  || delta > 0.02)
                {
                    ROS_INFO("time: [%f] ", delta);
                    begin = ros::Time::now().toSec();
                    std::cout<<"now publishing"<<std::endl;
                    drone_pos_pub.publish(pose);
                }
            }
            takeoff_time_init  = true;
        }


        offb_set_mode.request.custom_mode = "OFFBOARD";
        arm_cmd.request.value = true;

        if( current_state.mode != "OFFBOARD" &&
            (ros::Time::now() - last_request > ros::Duration(5.0))){
            if( set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        } else {
            if( !current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(5.0))){
                if( arming_client.call(arm_cmd) &&
                    arm_cmd.response.success){
                    ROS_INFO("Vehicle armed");
                }
                last_request = ros::Time::now();
            }
        }
        if (current_pose.pose.position.z < pose.pose.position.z)
        {
            drone_pos_pub.publish(pose);
            return false;
        }
        else
        {
            std::cout<<"Took off"<<std::endl;
//            drone_pos_pub.publish(pose);
            return true;
        }

    }

    std::tuple<double, double, arma::mat>  get_center_moments(arma::mat points)
    {
        double sum;
        int m_00 = 4;

        // m_ij = k=1 sum to n ( (x_k)^i * (y_k)^j )
        float m_01 = points(0, 1) + points(1, 1) + points(2, 1)+ points(3, 1);
        float m_10 = points(0, 0) + points(1, 0) + points(2, 0)+ points(3, 0);

        // x_g = m_10/m_00  y_g = m_01/m_00;
        double x_g = m_10/m_00;
        double y_g = m_01/m_00;

        arma::mat centered_m_list = arma::zeros<arma::mat>(4,4);
        arma::mat pre_centered_m = arma::zeros<arma::mat>(4,2);

        pre_centered_m(arma::span(0,3), 0) = points(arma::span(0,3), 0) - x_g;
        pre_centered_m(arma::span(0,3), 1) = points(arma::span(0,3), 1) - y_g;
        pre_centered_m.print("pre:");
        for(size_t i=0; i<4; i++)
        {
            for(size_t j=0; j<4; j++)
            {
                sum = 0;
                for(size_t k=0; k<4; k++)
                {
                    sum += (pow(pre_centered_m(k, 0), i) *  pow(pre_centered_m(k, 1), j));
//                      sum += i * log(abs(pre_centered_m(k, 0))) +  j* log(abs(pre_centered_m(k, 1)));
//                    std::cout<<"i="<<log(pre_centered_m(k, 0))<<" j="<<log(pre_centered_m(k, 1))<<std::endl;
                }
                centered_m_list(i, j) = sum;
            }
        }
        /*
        pre_centered_m.print("x-xg");
        centered_m_list.print("centered moments: ");
        std::cout<<"x_g="<<x_g<<" y_g="<<y_g<<" m_01="<< m_01<<" m_10="<<m_10<<std::endl;*/
        return std::make_tuple(x_g, y_g, centered_m_list) ;
    }

    std::tuple<double, double, arma::mat, arma::mat,
                double, double, double>compute_jacobian(arma::mat center_moments_d
                          ,arma::mat center_moments, double x_g, double y_g, double xd_g, double yd_g)
    {
        int m_00, beta, gamma;
        float z_d, a_d, a_c, a_n, x_n, y_n, eta_31, eta_32, eta_11, eta_12
              , eta_21, eta_22, d, alpha_wx, alpha_wy;
        double alpha_d, alpha;

        arma::mat J_moments =  -1 * arma::diagmat (arma::ones<arma::vec>(4));
        arma::mat J_moments_d =  -1 * arma::diagmat (arma::ones<arma::vec>(4));

        arma::mat n_moments_d = arma::zeros<arma::mat>(4,4);
        arma::mat n_moments = arma::zeros<arma::mat>(4,4);

        m_00 = 4;
        n_moments_d = center_moments_d / m_00;
        n_moments = center_moments / m_00;

        // a = u_20 + u_02
        a_d = center_moments_d(2,0) + center_moments_d(0, 2);
        a_c = center_moments(2,0) + center_moments(0, 2);

        z_d = 1;
        // a_n = z_d * sqrt( a_d / a_c)
        a_n = z_d * sqrt( a_d / a_c);

        // x_normalized = x_n = a_n * x_g   y_n = a_n * y_g
        x_n = a_n * x_g;
        y_n = a_n * y_g;

        float xd_n = a_n * xd_g;
        float yd_n = a_n * yd_g;

        // alpha = 0.5 * arctan(2 * u_11/ (u_20 - u_02))

        alpha_d = 0.5 * atan((2*center_moments_d(1,1))/(center_moments_d(2,1) - center_moments_d(1,2)));
        if (isnan(alpha_d)) alpha_d = 0;
        alpha = 0.5 * atan((2*center_moments(1,1))/(center_moments(2,1) - center_moments(1,2)));
        std::cout<<"alpha= "<<alpha<<std::endl;
        std::cout<<"alpha_d= "<<alpha_d<<std::endl;
        std::cout<<"area= "<<a_c<<"area_d= "<<a_d<<std::endl;
        eta_31 = y_g + ((y_g * center_moments(0,2) + x_g * center_moments(1,1) + center_moments(2,1) + center_moments(0,3)) / a_c);
        eta_32 = x_g + ((x_g * center_moments(2,0) + y_g * center_moments(1,1) + center_moments(1,2) + center_moments(3,0)) / a_c);
        eta_11 = n_moments(1,1) + x_g*(y_g - eta_31);
        eta_12 = n_moments(2,0) + x_g*(x_g - eta_32);
        eta_21 = n_moments(0,2) + y_g*(y_g - eta_31);
        eta_22 = n_moments(1,1) + y_g*(x_g - eta_32);

        beta = 4;
        gamma = 2;

        d = pow((center_moments(2,0) - center_moments(0,2)), 2) + 4 * pow(center_moments(1,1), 2);

        alpha_wx = (beta*(center_moments(1,2)*(center_moments(2,0) - center_moments(0,2))
                      + center_moments(1,1) * (center_moments(0,3) - center_moments(2,1)))
                + gamma * x_g *(center_moments(0,2) *(center_moments(2,0) - center_moments(0,2))
                  -2* pow(center_moments(1,1), 2)) + gamma* y_g * center_moments(1,1) * (center_moments(2,0) + center_moments(0,2)))/d;

        alpha_wy = (beta*(center_moments(2,1)*(center_moments(0,2) - center_moments(2,0))
                      + center_moments(1,1) * (center_moments(3,0) - center_moments(1,2)))
                + gamma * (center_moments(2,0) *(center_moments(0,2) - center_moments(2,0))
                  -2* pow(center_moments(1,1), 2)) + gamma* x_g * center_moments(1,1) * (center_moments(2,0) + center_moments(0,2)))/d;

//        arma::rowvec v =  {-1, 0, 0, y_n};
//        J_moments(0 , arma::span(0,3)) = v ;
//        v = {0, -1, 0, -x_n};
//        J_moments(1 , arma::span(0,3)) = v ;
//        v = {0, 0, -1, 0};
//        J_moments(2 , arma::span(0,3)) = v ;
//        v = {0, 0, 0, -1};
//        J_moments(3 , arma::span(0,3)) = v ;
        J_moments(0,3) = y_n;
        J_moments(1,3) = -x_n;

        J_moments_d(0,3) = yd_n;
        J_moments_d(1,3) = -xd_n;

//        J_moments.print("trasa");

        return std::make_tuple(alpha, alpha_d, J_moments, J_moments_d, a_n, x_n, y_n) ;
    }
    void compute_ctrl_law()
    {
        geometry_msgs::Twist msg;
        // ROS_INFO("%d %d %d",c_flag, k_flag, m_flag );
        if(c_flag && k_flag && m_flag)
        {
            w = 50;
            float h = 100/2;
            float f_x = K(0);

            float f_y = K(4);

            float u_0 = K(2);
            float v_0 = K(5);
//            arma::mat p_desired = {{240, 60}, {240, 180}, {80, 180}, {80, 60}};
//            arma::mat p_desired = {{365.7, 156.7}, {365, 317.5}, {205.6, 317}, {205.5, 158.3}};
            arma::mat p_current = {{this->fid->x0, this->fid->y0}, {this->fid->x1, this->fid->y1},
                                   {this->fid->x2, this->fid->y2}, {this->fid->x3, this->fid->y3}};
            arma::mat p_desired2 = {{u_0 + w, v_0 - h}, {u_0 + w, v_0 + h}, {u_0 - w, v_0 + h}, {u_0 - w, v_0 - h}};

            // p_current.print("p");
//            arma::mat e = p_current - p_desired;

//            arma::mat J = arma::zeros<arma::mat>(8,6);
            arma::mat J = arma::zeros<arma::mat>(8,6);

            p_desired = {{u_0 - w, v_0 - h}, {u_0 + w, v_0 - h}, {u_0 + w, v_0 + h}, {u_0 - w, v_0 + h}} ;

            float z_c = this->ft->transform.translation.z;
            if (z_c == 0) z_c =1;
//            float lambda = 0.1;
//            float lambda = 0.1;
            int row_idx = 0;
//            arma::mat e = arma::zeros<arma::mat>(4,2);
            arma::mat e = arma::zeros<arma::mat>(4,2);


//            for (size_t i=0; i<4; i++)
            for (size_t i=0; i<4; i++)

            {
                float x_c = (p_current(i, 0)- u_0)/f_x;
                float y_c = (p_current(i, 1)- v_0)/f_y;
                float x_d = (p_desired(i, 0)- u_0)/f_x;
                float y_d = (p_desired(i, 1)- v_0)/f_y;
                p_desired2(i,0) = x_d;
                p_desired2(i,1) = y_d;
                e(i, 0) = x_c - x_d;
                e(i, 1) = y_c - y_d;
                p_current(i,0) = x_c;
                p_current(i,1) = y_c;
//                std::cout<<"x_c"<<x_c<<y_c<<x_d<<y_d<<" "<<u_0<<" "<<v_0<<" "<<f_x<<std::endl;
            }
            e = p_current - p_desired;

            double x_g, xd_g, x_n;
            double y_g, yd_g, y_n;
            arma::mat centered_m_list, centered_m_list_d;
            std::tie(xd_g, yd_g, centered_m_list_d) = get_center_moments(p_desired2);
            std::cout<<x_g<<y_g;
            centered_m_list_d.print("k");
            std::tie(x_g, y_g, centered_m_list) =get_center_moments(p_current);
            std::cout<<x_g<<y_g;
            centered_m_list.print("k");
            arma::mat J_moments_d, J_moments, J_combined;
            double alpha, alpha_d, a_n;
            std::tie(alpha, alpha_d, J_moments, J_moments_d, a_n, x_n, y_n) = compute_jacobian(centered_m_list_d, centered_m_list, x_g, y_g, xd_g, yd_g);
            double z_d = 1;
            J_combined = 0.5 * (J_moments + J_moments_d);
            J_combined.print("J_comb:");
            arma::mat J_pinv = pinv(J_combined);
            arma::vec e_new = {x_n - (z_d *xd_g), y_n - (z_d *yd_g), a_n - z_d,  alpha - alpha_d };
            float lambda = 0.5;
//            J_pinv.print("J_pinv");
            arma::mat v_c = - lambda * J_pinv * e_new;
            v_c.print("VC:");

            for (size_t i=0; i<4; i++)
            {
                float x_c = (p_current(i, 0)- u_0)/f_x;
                float y_c = (p_current(i, 1)- v_0)/f_y;
                float u_bar = p_current(i, 0);
                float v_bar = p_current(i, 1);
                arma::mat Jp = {{-f_x/z_c, 0, u_bar/z_c, (u_bar*v_bar)/f_x, -(f_x+ (u_bar*u_bar / f_x)), v_bar},
                                {0, -f_y/z_c, v_bar/z_c, f_y+ (v_bar*v_bar / f_y), -(u_bar*v_bar)/f_y,  -u_bar}};
//                arma::mat Jp = {{-1/z_c, 0, x_c/z_c, x_c*y_c, -(1+ (x_c*x_c)), y_c},
//                                {0, -1/z_c, y_c/z_c, 1+ (y_c*y_c), -(x_c*y_c),  -x_c}};
                if (row_idx == 0)
                {
                    J(arma::span(i,i+1), arma::span(0,5)) = Jp;
                }
                else
                {
                    J(arma::span(row_idx,row_idx+1), arma::span(0,5)) = Jp;
                }
                row_idx += 2;
            }
//            J.print("J:");
            ROS_INFO("ZZZZZ  : [%f] ", z_c);
            // e.print("e:");
            // J.print();
//            arma::mat e_reshaped = arma::zeros<arma::mat>(8,1);
            arma::mat e_reshaped = arma::zeros<arma::mat>(8,1);
            row_idx = 0;
            for (size_t i=0; i<4; i++)
            {
                e_reshaped(row_idx,0) = e(i,0);
                e_reshaped(row_idx+1,0) = e(i,1);
                row_idx += 2;
            }
//            e_reshaped.print("e:");
            J_pinv = arma::pinv(J);
//            J_pinv.print("JJ:");
            double norm = arma::norm(e_reshaped, 2);
//            std::cout<<"norm: "<<norm<<std::endl;
//            if(norm < 150)
//            {
//                lambda = 0.3;
//            }

//            arma::mat v_c = - lambda * J_pinv * e_reshaped;
//            if(std::abs(v_c(0)) > 0.8)
//            {
//                if (v_c(0)<0)
//                {
//                    v_c(0) = -0.8;
//                }
//                else{v_c(0) = 0.8;}

//            }
//            if(std::abs(v_c(1)) > 0.8)
//            {
//                if (v_c(1)<0)
//                {
//                    v_c(1) = -0.8;
//                }
//                else{v_c(1) = 0.8;}

//            }
//            if(std::abs(v_c(2)) > 0.8)
//            {
//                if (v_c(2)<0)
//                {
//                    v_c(2) = -0.8;
//                }
//                else{v_c(2) = 0.8;}

//            }
//            if(std::abs(v_c(5)) > 0.8)
//            {
//                if (v_c(5)<0)
//                {
//                    v_c(5) = -0.8;
//                }
//                else{v_c(5) = 0.8;}

//            }
//            v_c.print("vc:");
            //TODO Publish rate 50 hz
            m_flag = false;
            c_flag = false;

            delta = ros::Time::now().toSec() - begin;
            if (!time_init || delta > 0.02)
            {
                ROS_INFO("time: [%f] ", delta);
                std::cout<<"Now velocity control"<<std::endl;
                begin = ros::Time::now().toSec();
                time_init = true;
//                pub.publish(msg);
                vel_cnt_full.coordinate_frame = 8;
                vel_cnt_full.type_mask = 1991; //Ignore everything except the linear x,y,z velocities and angular yaw
                vel_cnt_full.velocity.x =  - v_c(1);
                vel_cnt_full.velocity.y = - v_c(0);
                vel_cnt_full.velocity.z = - v_c(2);
                vel_cnt_full.yaw_rate =  -v_c(3);

                vel_cnt_full.header.stamp.sec = ros::Time::now().toSec();
                vel_cnt_full.header.stamp.nsec = ros::Time::now().toNSec();
                vel_cnt_full.header.frame_id = "base_link";
                drone_vel_pub.publish(vel_cnt_full);
//                geometry_msgs::PoseStamped pose;

//                pose.pose.position.x = 0;
//                pose.pose.position.y = 0;
//                pose.pose.position.z = 1.6;

//                drone_pos_pub.publish(pose);

            }
            
        }
    }

};


int main(int argc, char **argv)
    {

        ros::init(argc, argv, "sub_ibvs");
        ros::NodeHandle nh;
        
        IBVS ibvs = IBVS(&nh);
        // ros::spin();
        while (ros::ok())
        {
            if (!ibvs.takeoff)
            {
                ibvs.takeoff = ibvs.drone_takeOff();
            }

            if (ibvs.takeoff)
            {
//                ibvs.takeoff = ibvs.drone_takeOff();
                ibvs.compute_ctrl_law();
            }
//            std::cout<<res;

            ros::spinOnce();
        }
        return 0;
    }
