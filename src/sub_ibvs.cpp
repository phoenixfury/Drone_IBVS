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
#include <mavros_msgs/CommandLong.h>
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
    fiducial_msgs::Fiducial fid_val[2];    //An aruco marker
    fiducial_msgs::Fiducial * fid = fid_val;    //An aruco marker
    fiducial_msgs::FiducialTransform ft_val; //An aruco marker transform
    fiducial_msgs::FiducialTransform * ft = &ft_val; //An aruco marker transform

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
    ros::ServiceClient force_arming_client;
    //Variables
    mavros_msgs::SetMode offb_set_mode;
    mavros_msgs::CommandBool arm_cmd;
    mavros_msgs::CommandLong force_arm_cmd;
    mavros_msgs::PositionTarget vel_cnt_full;

    // Visualization
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;

    // NEED OPTIMIZATION
    int current_markers = 0;
    float x_g, y_g, xd_g, yd_g, x_n, y_n, a_d, a_n;
    arma::mat centered_m_list = arma::zeros<arma::mat>(4,4);
    arma::mat pre_centered_m = arma::zeros<arma::mat>(4,2);

    arma::mat centered_m_list_d = arma::zeros<arma::mat>(4,4);
    arma::mat pre_centered_m_d = arma::zeros<arma::mat>(4,2);

    arma::mat J_combined =  -1 * arma::diagmat (arma::ones<arma::vec>(4));
    arma::mat J_moments =  -1 * arma::diagmat (arma::ones<arma::vec>(4));
    arma::mat J_moments_d =  -1 * arma::diagmat (arma::ones<arma::vec>(4));
    arma::mat J_pinv;

    arma::mat n_moments_d = arma::zeros<arma::mat>(4,4);
    arma::mat n_moments = arma::zeros<arma::mat>(4,4);

    arma::vec err;
    // Need optimization
    double norm_err_max = 1;
    float w = 75;
    float h = 75;
    float gain;
//    float w = 50;
//    float h = 100/2;
    bool marker1 = true;
    arma::mat v={0, 0, 0, 0};
    arma::mat v2={0, 0, 0, 0};
    arma::mat p_desired = {{480/w, 128/w}, {480/w, 384/w}, {160/w, 384/w}, {160/w, 128/w}};
//    arma::mat p_desired = {{480/w, 128/w}, {480/w, 384/w}, {160/w, 384/w}, {160/w, 128/w}};
    arma::mat p_desired_transformed = {{480/w, 128/w}, {480/w, 384/w}, {160/w, 384/w}, {160/w, 128/w}};

    float f_x;

    float f_y;

    float u_0;
    float v_0;
    // Camera intrinsics
    arma::mat K = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Initialization flags
    bool k_flag = false;
    bool c_flag = false;
    bool m_flag = false;
    bool takeoff_time_init = false;
    bool time_init = false;

    bool first_detection = false;

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
        // Jackal velocity controller
//        pub = nh->advertise<geometry_msgs::Twist>("jackal_velocity_controller/cmd_vel", 1000);

        // Aruco marker detections
        vertices_sub = nh->subscribe("fiducial_vertices", 3, &IBVS::markerCallback, this);
        transforms_sub = nh->subscribe("fiducial_transforms", 3, &IBVS::transformCallback, this);

        //Camera Info
        camera_info_sub = nh->subscribe("/camera/rgb/camera_info", 1, &IBVS::c_infoCallback, this);

        //Drone publishers and subscribers
        drone_pos_pub =nh->advertise<geometry_msgs::PoseStamped>("mavros/setpoint_position/local", 5);
        drone_vel_pub =nh->advertise<mavros_msgs::PositionTarget>("mavros/setpoint_raw/local", 5);
        drone_state_sub = nh->subscribe<mavros_msgs::State>("mavros/state", 5, &IBVS::drone_state_callback, this);
        drone_pose_sub =nh->subscribe<geometry_msgs::PoseStamped>("mavros/local_position/pose", 5, &IBVS::drone_pose_callback, this);

        //Drone services
        arming_client = nh->serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
        force_arming_client = nh->serviceClient<mavros_msgs::CommandLong>("mavros/cmd/command");
        set_mode_client = nh->serviceClient<mavros_msgs::SetMode>("mavros/set_mode");

        // Visualization advertiser and subscriber
        image_sub_ = it_.subscribe("/fiducial_images", 2, &IBVS::imageCb, this);
        image_pub_ = it_.advertise("/sub_ibvs/output_video", 2);

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
            this->f_x = K(0);

            this->f_y = K(4);

            this->u_0 = K(2);
            this->v_0 = K(5);

            this->p_desired = {{this->u_0 - this->w, this->v_0 + this->h}, {this->u_0 - this->w, this->v_0 - this->h},
                               {this->u_0 + this->w, this->v_0 - this->h}, {this->u_0 + this->w, this->v_0 + this->h}} ;

            this->init_desired_variables(this->p_desired, 0);

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

        // Draw the 4 desired corners on the video stream
        if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
          cv::circle(cv_ptr->image, cv::Point(p_desired(0,0),p_desired(0,1)), 3, CV_RGB(0, 0, 255), -1);
          cv::circle(cv_ptr->image, cv::Point(p_desired(1,0),p_desired(1,1)), 3, CV_RGB(0, 128, 0), -1);
          cv::circle(cv_ptr->image, cv::Point(p_desired(2,0),p_desired(2,1)), 3, CV_RGB(0, 128, 0), -1);
          cv::circle(cv_ptr->image, cv::Point(p_desired(3,0),p_desired(3,1)), 3, CV_RGB(0, 128, 0), -1);

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
        this->current_markers = fva.fiducials.size();
        for (size_t i=0; i<fva.fiducials.size(); i++) {
            this->fid[i].fiducial_id = fva.fiducials[i].fiducial_id;
            this->fid[i].x0 = fva.fiducials[i].x0;
            this->fid[i].y0 = fva.fiducials[i].y0;
            this->fid[i].x1 = fva.fiducials[i].x1;
            this->fid[i].y1 = fva.fiducials[i].y1;
            this->fid[i].x2 = fva.fiducials[i].x2;
            this->fid[i].y2 = fva.fiducials[i].y2;
            this->fid[i].x3 = fva.fiducials[i].x3;
            this->fid[i].y3 = fva.fiducials[i].y3;
        }
        m_flag = true;
    }


    void transformCallback(const fiducial_msgs::FiducialTransformArray fta)
    {
        for (size_t i=0; i<fta.transforms.size(); i++) {

            this->ft->fiducial_id = fta.transforms[i].fiducial_id;

            this->ft->transform = fta.transforms[i].transform;
        }
        c_flag = true;
    }


    bool drone_takeOff(float x, float y, float h)
    {
        geometry_msgs::PoseStamped pose;

        pose.pose.position.x = x;
        pose.pose.position.y = y;
        pose.pose.position.z = h;

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
            (ros::Time::now() - last_request > ros::Duration(2.0))){
            if( set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        } else {
            if( !current_state.armed &&
                (ros::Time::now() - last_request > ros::Duration(2.0))){
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


    void init_desired_variables(arma::mat points, float z)
    {
        // Initialize the constant matrices and variables
        this->p_desired = points;

        int m_00 = 4;

        for (size_t i=0; i<4; i++)

        {
            float x_d = (this->p_desired(i, 0)- this->u_0)/this->f_x;
            float y_d = (this->p_desired(i, 1)- this->v_0)/this->f_y;

            this->p_desired_transformed(i,0) = x_d;
            this->p_desired_transformed(i,1) = y_d;
        }
        get_center_moments(this->p_desired_transformed);
        this->xd_g = this->x_g;
        this->yd_g = this->y_g;
        this->centered_m_list_d = this->centered_m_list;

        this->n_moments_d = this->centered_m_list_d / m_00;

        this->a_d = this->centered_m_list_d(2,0) + this->centered_m_list_d(0, 2);

    }

    void get_center_moments(arma::mat points)
    {
        double sum;
        int m_00 = 4;

        // m_ij = k=1 sum to n ( (x_k)^i * (y_k)^j )
        float m_01 = points(0, 1) + points(1, 1) + points(2, 1)+ points(3, 1);
        float m_10 = points(0, 0) + points(1, 0) + points(2, 0)+ points(3, 0);

        // x_g = m_10/m_00  y_g = m_01/m_00;
        this->x_g = m_10/m_00;
        this->y_g = m_01/m_00;

        this->pre_centered_m(arma::span(0,3), 0) = points(arma::span(0,3), 0) - this->x_g;
        this->pre_centered_m(arma::span(0,3), 1) = points(arma::span(0,3), 1) - this->y_g;

        for(size_t i=0; i<4; i++)
        {
            for(size_t j=0; j<4; j++)
            {
                sum = 0;
                for(size_t k=0; k<4; k++)
                {
                    sum += (pow(this->pre_centered_m(k, 0), i) *  pow(this->pre_centered_m(k, 1), j));
                }
                this->centered_m_list(i, j) = sum;
            }
        }
    }


    float compute_jacobian(float z_d)
    {
        int m_00= 4;
        float a_c;

        this->n_moments = this->centered_m_list / m_00;

        // a = u_20 + u_02

        a_c = this->centered_m_list(2,0) + this->centered_m_list(0, 2);

        // a_n = z_d * sqrt( a_d / a_c)
        this->a_n = z_d * sqrt( this->a_d / a_c);

        // x_normalized = x_n = a_n * x_g   y_n = a_n * y_g
        this->x_n = this->a_n * this->x_g;
        this->y_n = this->a_n * this->y_g;

        std::cout<<"area= "<<a_c<<"area_d= "<<this->a_d<<std::endl;

        this->J_moments(0,3) = y_n;
        this->J_moments(1,3) = -x_n;

        this->J_moments_d(0,3) = this->a_n * this->xd_g;
        this->J_moments_d(1,3) = - (this->a_n * this->xd_g);
        return a_n;
    }

    std::tuple<double, arma::mat> compute_ctrl_law(fiducial_msgs::Fiducial current_fid, float z_dd)
    {
//            arma::mat p_desired = {{365.7, 156.7}, {365, 317.5}, {205.6, 317}, {205.5, 158.3}};
            arma::mat p_current = {{current_fid.x0, current_fid.y0}, {current_fid.x1, current_fid.y1},
                                   {current_fid.x2, current_fid.y2}, {current_fid.x3, current_fid.y3}};

//            p_desired = {{this->u_0 - this->w, this->v_0 - this->h}, {this->u_0 + this->w, this->v_0 - this->h},
//                         {this->u_0 + this->w, this->v_0 + this->h}, {this->u_0 - this->w, this->v_0 + this->h}} ;


            float z_c = this->ft->transform.translation.z;

            if (z_c == 0) z_c =1;


            for (size_t i=0; i<4; i++)

            {
                float x_c = (p_current(i, 0)- u_0)/f_x;
                float y_c = (p_current(i, 1)- v_0)/f_y;

                p_current(i,0) = x_c;
                p_current(i,1) = y_c;
            }


            get_center_moments(p_current);

            float a_n;

            a_n = compute_jacobian(z_dd);

            float alpha = atan2(p_current(2,1)-p_current(0,1), p_current(2,0) - p_current(0,0));
            float alpha_d = atan2(this->p_desired_transformed(2,1)-this->p_desired_transformed(0,1), this->p_desired_transformed(2,0) - this->p_desired_transformed(0,0));

            float z_d = z_dd;
//            double z_d = 1;
            this->J_combined = 0.5 * (this->J_moments + this->J_moments_d);
            this->J_combined.print("J_comb:");

            this->J_pinv = pinv(this->J_combined);
            this->err = {this->x_n - (z_d *this->xd_g), this->y_n - (z_d *this->yd_g), a_n - z_d, alpha - alpha_d };

            std::cout<<"alpha: "<<alpha<<"alphad: "<<alpha_d<<std::endl;
            if (!first_detection)
            {
                norm_err_max = arma::norm(this->err, 2);
                first_detection = true;
            }
            float norm_err = arma::norm(this->err, 2);

            float min_lambda =1.5;
            float max_lambda = 0.5;
            float lambda = (max_lambda - min_lambda) * (norm_err / norm_err_max) + min_lambda;

//            J_pinv.print("J_pinv");
            if(this->marker1)
            {
                this->v = - lambda* this->J_pinv * this->err;

            }
            else
            {
                this->v = -lambda* this->J_pinv * this->err;
            }
            this->v.print("VC:");

            ROS_INFO("Altitude: [%f] ", z_c);


            std::cout<<"error norm: "<<norm_err<<std::endl;

            //TODO Publish rate 50 hz
            m_flag = false;
            c_flag = false;

            return std::make_tuple(norm_err, this->v);

    }


    void send_ctrl_signal(arma::mat v_c, float er)
    {

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
    }


    void marker_selection()
    {
        if (c_flag && k_flag && m_flag) {
            begin = ros::Time::now().toSec();
            float threshold = 0.06;
            float z = 0.6;
            float z_2 = 0.2;


            float er;
            for (size_t i = 0; i < this->current_markers; i++) {
                if (this->marker1 && this->fid[i].fiducial_id == 0) {
                    std::tie(er, this->v) = compute_ctrl_law(this->fid[i], z);
                    if (er < threshold) { this->marker1 = false;
                        this->w = 40;
                        this->h = 40;
                        this->p_desired = {{this->u_0 - this->w, this->v_0 + this->h}, {this->u_0 - this->w, this->v_0 - this->h},
                                           {this->u_0 + this->w, this->v_0 - this->h}, {this->u_0 + this->w, this->v_0 + this->h}} ;
                        init_desired_variables( this->p_desired,z_2);}
                }

                if (!this->marker1 && this->fid[i].fiducial_id == 7)
                {
                    std::tie(er, this->v) = compute_ctrl_law(this->fid[i], z_2);
                    arm_cmd.request.value = false;
                    this->force_arm_cmd.request.command = 400;
                    this->force_arm_cmd.request.confirmation = 0;
                    this->force_arm_cmd.request.param1 = 0;
                    this->force_arm_cmd.request.param1 = 21196;
                    this->force_arm_cmd.request.broadcast = false;

                    if( current_state.armed && er < 0.015)
                        {
                            arming_client.call(arm_cmd);
                            if( this->force_arming_client.call(force_arm_cmd))
                            {
                                ROS_INFO("Vehicle disarmed");
                                ros::shutdown();
                            }
                        }
                }



            }
            send_ctrl_signal(this->v, er);
            std::cout<<"time: "<<ros::Time::now().toSec()- begin<<std::endl;

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
                ibvs.takeoff = ibvs.drone_takeOff(0.175, 0.175, 1.25);
            }

            if (ibvs.takeoff)
            {
                ibvs.marker_selection();
            }

            ros::spinOnce();
        }
        return 0;
    }
