#include "ros/ros.h"

#include "obj_db_msgs/RegisterSegment.h"    //RegisterSegment srv type
#include "sensor_msgs/PointCloud2.h"

//pcl_general
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/impl/point_types.hpp>
//spin
#include <pcl/features/spin_image.h>
//shot
#include <pcl/features/shot.h>
#include <pcl/keypoints/iss_3d.h>
//#include "shot_omp.h" --multithread?
//fpfh
#include <pcl/features/fpfh.h>
//pcl visualize
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
//corresponadance grouping
#include <pcl/correspondence.h>
#include <pcl/recognition/cg/geometric_consistency.h>

//TODO:
// - decide whether vpp msgs needed if not remove dependency

class ObjectSegment{
    public:
        uint16_t _id;   //local library id
        uint16_t _label;
        pcl::PointCloud<pcl::PointXYZ>::Ptr _pc;
        pcl::PointCloud<pcl::FPFHSignature33>::Ptr _signature;
        pcl::PointCloud<pcl::PointXYZ>::Ptr _keypoints;
        // geometry_msgs::Pose pose;
        // geometry_msgs::Vector3 dimensions;
        // uint16_t _id;
        // std::string _label;

        ObjectSegment(uint16_t id, uint16_t label, pcl::PointCloud<pcl::PointXYZ>::Ptr pc,
                      pcl::PointCloud<pcl::FPFHSignature33>::Ptr signature, 
                      pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints){
            _id = id;
            _label = label;
            _pc = pc;
            _signature = signature;
            _keypoints = keypoints;
        }

        // std::string to_string(){
        //     ROS_INFO_STREAM("Object id: " << _id << " | category: " << _label
        //         << "pose: " << pose);
        //     return "Object id: " + std::to_string(_id) + " | category: " + _label;
        // }
};

class ObjectDB{
    public:
        std::vector<ObjectSegment*> segment_library;
        std::vector<uint16_t> tracked_ids;

        ObjectDB(){}

        ~ObjectDB(){
            std::cout << "---- ObjectDB - segments in library: ---- \n";
            for(ObjectSegment* obj : segment_library){
                std::cout << "id: "<< obj->_id << " label: "<< obj->_label <<"\n";
            }
        }
        
        void ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in, 
                 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out){
            

        }

        double computeICPScore(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
                               pcl::PointCloud<pcl::PointXYZ>::Ptr target){
            pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
            icp.setInputSource(cloud_in);
            icp.setInputTarget(target);    //rather pointer here?
            pcl::PointCloud<pcl::PointXYZ> Final;
            icp.align(Final);
            double score = icp.getFitnessScore();
            
            std::cout << "has converged:" << icp.hasConverged() << " score: " <<
                score << std::endl;
            //std::cout << icp.getFinalTransformation() << std::endl;
            return score;
        }


        //(input_pointcloud, output_SHOT_descriptors)
        void computeShotDescriptors(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                    pcl::PointCloud<pcl::SHOT352>::Ptr descriptors){
            
            // Object for storing the normals.
	        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

            //keypoints, downsample
            pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
            // ISS keypoint detector object.
            pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> detector;
            detector.setInputCloud(cloud);
            pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
            detector.setSearchMethod(kdtree);
            double resolution = computeCloudResolution(cloud);
            // Set the radius of the spherical neighborhood used to compute the scatter matrix.
            detector.setSalientRadius(6 * resolution);
            // Set the radius for the application of the non maxima supression algorithm.
            detector.setNonMaxRadius(4 * resolution);
            // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
            detector.setMinNeighbors(5);
            // Set the upper bound on the ratio between the second and the first eigenvalue.
            detector.setThreshold21(0.975);
            // Set the upper bound on the ratio between the third and the second eigenvalue.
            detector.setThreshold32(0.975);
            // Set the number of prpcessing threads to use. 0 sets it to automatic.
            detector.setNumberOfThreads(4);
            
            detector.compute(*keypoints);
            std::cout << "keypoints cloud size:" << keypoints->size() << "\n";
            
            
            // Estimate the normals.
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
            normalEstimation.setInputCloud(keypoints);
            normalEstimation.setRadiusSearch(0.03);
            //TODO: kdtree redeclaration?
            pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointXYZ>);
            normalEstimation.setSearchMethod(kdtree2);
            normalEstimation.compute(*normals);

            // SHOT estimation object.
            pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
            shot.setInputCloud(keypoints);
            shot.setInputNormals(normals);
            // The radius that defines which of the keypoint's neighbors are described.
            // If too large, there may be clutter, and if too small, not enough points may be found.
            shot.setRadiusSearch(0.02);

            shot.compute(*descriptors);
        }

        double computeShotScore(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr target){
            // Objects for storing the SHOT descriptors for the input and model
            pcl::PointCloud<pcl::SHOT352>::Ptr desc_in(new pcl::PointCloud<pcl::SHOT352>());
            pcl::PointCloud<pcl::SHOT352>::Ptr desc_model(new pcl::PointCloud<pcl::SHOT352>());
         
            computeShotDescriptors(cloud_in, desc_in);
            std::cout << "shot descriptor size:" << desc_in->size() <<"\n";
            computeShotDescriptors(target, desc_model);

            // A kd-tree object that uses the FLANN library for fast search of nearest neighbors.
            pcl::KdTreeFLANN<pcl::SHOT352> matching;
            matching.setInputCloud(desc_model);
            // A Correspondence object stores the indices of the query and the match,
            // and the distance/weight.
            pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());

            // Check every descriptor computed for the scene.
            for (size_t i = 0; i < desc_in->size(); ++i)
            {
                std::vector<int> neighbors(1);
                std::vector<float> squaredDistances(1);
                // Ignore NaNs.
                if (pcl_isfinite(desc_in->at(i).descriptor[0]))
                {
                    // Find the nearest neighbor (in descriptor space)...
                    int neighborCount = matching.nearestKSearch(desc_in->at(i), 1, neighbors, squaredDistances);
                    // ...and add a new correspondence if the distance is less than a threshold
                    // (SHOT distances are between 0 and 1, other descriptors use different metrics).
                    if (neighborCount == 1 && squaredDistances[0] < 0.25f)
                    {
                        pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i), squaredDistances[0]);
                        correspondences->push_back(correspondence);
                    }
                }
            }
            ROS_INFO_STREAM("Found " << std::to_string(correspondences->size()) 
                            << " correspondences." << std::endl);

            return 0.5;
        }

            //feature based registration: 
            //keypoints - SIFT?
            //descriptor
            //estimate correspondencies
            //reject bad correspondancies?
        typedef pcl::Histogram<153> SpinImage;
        void computeSpinImage(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                              pcl::PointCloud<SpinImage>::Ptr descriptors){
            
           // Object for storing the normals.
	        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

            //keypoints, downsample
            pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
            // ISS keypoint detector object.
            pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> detector;
            detector.setInputCloud(cloud);
            pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
            detector.setSearchMethod(kdtree);
            double resolution = computeCloudResolution(cloud);
            // Set the radius of the spherical neighborhood used to compute the scatter matrix.
            detector.setSalientRadius(6 * resolution);
            // Set the radius for the application of the non maxima supression algorithm.
            detector.setNonMaxRadius(4 * resolution);
            // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
            detector.setMinNeighbors(5);
            // Set the upper bound on the ratio between the second and the first eigenvalue.
            detector.setThreshold21(0.975);
            // Set the upper bound on the ratio between the third and the second eigenvalue.
            detector.setThreshold32(0.975);
            // Set the number of prpcessing threads to use. 0 sets it to automatic.
            detector.setNumberOfThreads(4);
            
            detector.compute(*keypoints);
            std::cout << "keypoints cloud size:" << keypoints->size() << "\n";
            
            // Estimate the normals.
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
            normalEstimation.setInputCloud(keypoints);
            normalEstimation.setRadiusSearch(0.03);
            //TODO: kdtree redeclaration?
            pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointXYZ>);
            normalEstimation.setSearchMethod(kdtree2);
            normalEstimation.compute(*normals);

            // Spin image estimation object.
            pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, SpinImage> si;
            si.setInputCloud(cloud);
            si.setInputNormals(normals);
            // Radius of the support cylinder.
            si.setRadiusSearch(0.02);
            // Set the resolution of the spin image (the number of bins along one dimension).
            // Note: you must change the output histogram size to reflect this.
            si.setImageWidth(8);

            si.compute(*descriptors);

        }

        double computeSpinScore(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in,
                                pcl::PointCloud<pcl::PointXYZ>::Ptr target){
             // Object for storing the spin image for each point.
	        pcl::PointCloud<SpinImage>::Ptr desc_in(new pcl::PointCloud<SpinImage>());
	        pcl::PointCloud<SpinImage>::Ptr desc_model(new pcl::PointCloud<SpinImage>());
            std::cout << "h1 \n";
            computeSpinImage(cloud_in, desc_in);
            computeSpinImage(target, desc_model);

            //compute euklidean distance between spinimg pointclouds -> 
             std::cout << "spin desc model size: " << desc_model->size() 
                       << " | spin cloud model size" << desc_in->size(); 
            //for(pcl::PointCloud<SpinImage>::iterator it 
            //    = cloud->begin(); it!= cloud->end(); it++){
            //    
            //}
        }
        
        
        
        void correspondenceGrouping(ObjectSegment *seg_in, ObjectSegment *obj_seg, 
                                    pcl::CorrespondencesPtr correspondences){
            
        
            std::vector<pcl::Correspondences> clusteredCorrespondences;
	        // Object for storing the transformations (rotation plus translation).
	        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transformations;
            // Note: here you would compute the correspondences.
            // It has been omitted here for simplicity.

            // Object for correspondence grouping.
            pcl::GeometricConsistencyGrouping<pcl::PointXYZ, pcl::PointXYZ> grouping;
            grouping.setSceneCloud(seg_in->_keypoints);
            grouping.setInputCloud(obj_seg->_keypoints);
            grouping.setModelSceneCorrespondences(correspondences);
            // Minimum cluster size. Default is 3 (as at least 3 correspondences
            // are needed to compute the 6 DoF pose).
            grouping.setGCThreshold(3);
            // Resolution of the consensus set used to cluster correspondences together,
            // in metric units. Default is 1.0.
            grouping.setGCSize(0.01);

            grouping.cluster(clusteredCorrespondences);

            //grouping.recognize(transformations, clusteredCorrespondences);

            // std::cout << "Model instances found: " << transformations.size() << std::endl << std::endl;
            // for (size_t i = 0; i < transformations.size(); i++)
            // {
            //     std::cout << "Instance " << (i + 1) << ":" << std::endl;
            //     std::cout << "\tHas " << clusteredCorrespondences[i].size() << " correspondences." << std::endl << std::endl;

            //     Eigen::Matrix3f rotation = transformations[i].block<3, 3>(0, 0);
            //     Eigen::Vector3f translation = transformations[i].block<3, 1>(0, 3);
            //     printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
            //     printf("\t\tR = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
            //     printf("\t\t    | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
            //     std::cout << std::endl;
            //     printf("\t\tt = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));
            // }
        }


        // This function by Tommaso Cavallari and Federico Tombari, taken from the tutorial
        // http://pointclouds.org/documentation/tutorials/correspondence_grouping.php
        double
        computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
        {
            double resolution = 0.0;
            int numberOfPoints = 0;
            int nres;
            std::vector<int> indices(2);
            std::vector<float> squaredDistances(2);
            pcl::search::KdTree<pcl::PointXYZ> tree;
            tree.setInputCloud(cloud);

            for (size_t i = 0; i < cloud->size(); ++i)
            {
                if (! pcl_isfinite((*cloud)[i].x))
                    continue;

                // Considering the second neighbor since the first is the point itself.
                nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
                if (nres == 2)
                {
                    resolution += sqrt(squaredDistances[1]);
                    ++numberOfPoints;
                }
            }
            if (numberOfPoints != 0)
                resolution /= numberOfPoints;

            return resolution;
        }

        void computeFPFHDescriptors(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                                    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors,
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints){
            	
                
            pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

            //keypoints - downsample
            //pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
            // ISS keypoint detector object.
            pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> detector;
            detector.setInputCloud(cloud);
            pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
            detector.setSearchMethod(kdtree);
            double resolution = computeCloudResolution(cloud);
            // Set the radius of the spherical neighborhood used to compute the scatter matrix.
            detector.setSalientRadius(6 * resolution);
            // Set the radius for the application of the non maxima supression algorithm.
            detector.setNonMaxRadius(4 * resolution);
            // Set the minimum number of neighbors that has to be found while applying the non maxima suppression algorithm.
            detector.setMinNeighbors(5);
            // Set the upper bound on the ratio between the second and the first eigenvalue.
            detector.setThreshold21(0.975);
            // Set the upper bound on the ratio between the third and the second eigenvalue.
            detector.setThreshold32(0.975);
            // Set the number of prpcessing threads to use. 0 sets it to automatic.
            detector.setNumberOfThreads(4);
            
            detector.compute(*keypoints);
            std::cout << "keypoints cloud size:" << keypoints->size() << "\n";

            // Estimate the normals.
            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
            normalEstimation.setInputCloud(keypoints);
            normalEstimation.setRadiusSearch(0.03);
            pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointXYZ>);
            normalEstimation.setSearchMethod(kdtree2);
            normalEstimation.compute(*normals);
            
            // FPFH estimation object.
            pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
            fpfh.setInputCloud(keypoints);
            fpfh.setInputNormals(normals);
            fpfh.setSearchMethod(kdtree);
            // Search radius, to look for neighbors. Note: the value given here has to be
            // larger than the radius used to estimate the normals.
            fpfh.setRadiusSearch(0.05);
            
            fpfh.compute(*descriptors);
            

        }   
        //Every descriptor in the scene should be matched against the descriptors of every model in the database, 
        //because this accounts for the presence of multiple instances of the model, which would not be recognized 
        //if we did it the other way around.
        double computeFPFHScore(ObjectSegment *seg_in, ObjectSegment *seg_model){

            // pcl::PointCloud<pcl::FPFHSignature33>::Ptr desc_in(new 
            //     pcl::PointCloud<pcl::FPFHSignature33>());
            // pcl::PointCloud<pcl::FPFHSignature33>::Ptr desc_model(new 
            //     pcl::PointCloud<pcl::FPFHSignature33>());             
            // std::cout << "computing FPFH descriptors \n";
            // computeFPFHDescriptors(cloud_in, desc_in);
            // computeFPFHDescriptors(target, desc_model);      
            pcl::PointCloud<pcl::FPFHSignature33>::Ptr desc_in = seg_in->_signature;
            pcl::PointCloud<pcl::FPFHSignature33>::Ptr desc_model = seg_in->_signature;

            // A kd-tree object that uses the FLANN library for fast search of nearest neighbors.
            pcl::KdTreeFLANN<pcl::FPFHSignature33> matching;
            matching.setInputCloud(desc_model);
            // A Correspondence object stores the indices of the query and the match,
            // and the distance/weight.
            pcl::CorrespondencesPtr correspondences(new pcl::Correspondences());
 
            // Check every descriptor computed for the scene.
            //std::cout << "squared distances: \n"; 
            for (size_t i = 0; i < desc_model->size(); ++i)
            {
                std::vector<int> neighbors(1);
                std::vector<float> squaredDistances(1);
    
            
                // Find the nearest neighbor (in descriptor space)...
                int neighborCount = matching.nearestKSearch(desc_in->at(i), 1, neighbors, squaredDistances);
                // ...and add a new correspondence if the distance is less than a threshold
                // (SHOT distances are between 0 and 1, other descriptors use different metrics).
                //std::cout << squaredDistances[0] << "|";
                if (neighborCount == 1 && squaredDistances[0] < 10.0f)
                {
                    pcl::Correspondence correspondence(neighbors[0], static_cast<int>(i), squaredDistances[0]);
                    correspondences->push_back(correspondence);
                }
            }  
            std::cout << "\n --> Found " << correspondences->size() << " correspondences." << std::endl;
            
            // pcl::visualization::PCLVisualizer viewer("Cloud Viewer");

            //at least 1/2 corresponding
            if((correspondences->size())*(correspondences->size())
                > ((seg_model->_signature.use_count() - correspondences->size())/2)*((seg_model->_signature.use_count() - correspondences->size())/2)
                /*seg_model->_signature.use_count() / 2*/){    
                // pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
                // viewer.addPointCloud (seg_in->_pc,"cloud in");
                // viewer.addPointCloud (seg_model->_pc,"prototype");
                // //viewer.runOnVisualizationThreadOnce(setBackground);
                // viewer.spin();
                return correspondences->size();

                //correspondenceGrouping(seg_in, seg_model, correspondences);
            }
            return 0.0;         
        }


        bool registerSegment(uint16_t label, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in){ 
            if(segment_library.size() > 0){
                double fitness[segment_library.size()];
                int i = 0; 
                //new ObjectSegment with signature
                pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>);;
                pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
                computeFPFHDescriptors(cloud_in, descriptors, keypoints);
                ObjectSegment* seg_in = new ObjectSegment(segment_library.size(), label, 
                    cloud_in, descriptors, keypoints);
                //compate to all stored  
                for(ObjectSegment* model : segment_library){
                    //fitness[i] = computeICPScore(cloud_in, seg->_pc);
                    //computeShotScore(cloud_in, seg->_pc);
                    //computeSpinScore(cloud_in, seg->_pc);
                    double val = computeFPFHScore(seg_in, model);
                    fitness[i] = val;
                    i++;
                }

                //find maximal fitness
                int fitmax = 0;
                int fitargmax = 0;
                for(int j = 0; j < segment_library.size(); j++){
                    if(fitness[j] > fitmax){
                        fitmax = fitness[j];
                        fitargmax = j;
                    }
                }
                if(fitmax > 1.0){
                    std::cout << "SEGMENT MATCHED with" << fitargmax;
                    pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
                    viewer.addPointCloud (seg_in->_pc,"cloud in");
                    viewer.addPointCloud (segment_library.at(fitargmax)->_pc,"prototype");
                    //viewer.runOnVisualizationThreadOnce(setBackground);
                    viewer.spin();   
                    
                    segment_library.push_back(seg_in);
                }else{
                    delete seg_in;
                }

            }else{//Library empty
                //create seg object and pass pointer to library
                if(label > 0){
                    //compute sign for the first one
                    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
                    computeFPFHDescriptors(cloud_in, descriptors, keypoints);
                    ObjectSegment* seg = new ObjectSegment(0, label, cloud_in, descriptors, keypoints);
                    segment_library.push_back(seg);
                    std::cout << "----model---- \n";
                    ROS_INFO_STREAM("segment 0 registered, label: " << std::to_string(label));
                    std::cout << "----model---- \n";
                    
                    // //pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
                    // pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
                    // //iewer.setBackgroundColor (1.0, 0.5, 1.0); 
                    // //viewer.showCloud (body,"body");
                    // viewer.showCloud (cloud_in,"cloud");

                    // while (!viewer.wasStopped ())
                    // {
                    // }
                    return true;
                }
            }
            
            return false;
        }

        //pair cloud with corresponding id
        bool registerSegment(obj_db_msgs::RegisterSegmentRequest& request,
                             obj_db_msgs::RegisterSegmentResponse& response){
            sensor_msgs::PointCloud2 pc = request.segment;
            //get semantic label of the first point
            int arrayPos = 0;
            int labelArrayPos  = arrayPos + pc.fields[8].offset;
            int semantic_label = 0;
            memcpy(&semantic_label, &pc.data[labelArrayPos], sizeof(int));
            
            ROS_INFO_STREAM("segment register request" << "- sem label:" 
                << std::to_string(semantic_label));

            //conversion to PCL pointcloud type
            pcl::PCLPointCloud2 pcl_pc;
            pcl_conversions::toPCL(pc, pcl_pc);
   
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromPCLPointCloud2(pcl_pc,*cloud_in);

            std::cout << "incoming cloud size:" << cloud_in->size() << "\n";
            registerSegment(semantic_label, cloud_in);

            return true;            
        }
};

int main(int argc, char **argv){
    ros::init(argc, argv, "obj_db");
    ros::NodeHandle nh;
    std::string param;

    ObjectDB oDB; 

    ros::ServiceServer seg_reg_service 
        = nh.advertiseService("register_segment", &ObjectDB::registerSegment, &oDB);

    ros::spin();    

    return 0;
}