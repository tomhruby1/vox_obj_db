#!/usr/bin/env python3

import rospy
import struct
import time
import sys

import tf2_ros
import tf
import tf_conversions
import geometry_msgs.msg
import tf2_geometry_msgs
import obj_db_msgs.srv

from tf.transformations import * 

from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

import numpy as np

from std_msgs.msg import String
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo
from std_msgs.msg import Header

import matplotlib.pyplot as plt

from m2dp import M2DP
import open3d as o3d
import copy

coco_labels = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'teennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

    
def get_point_cloud(seg):
    points = []
    for p in seg:
        points.append(p[0:3])
    mu = np.average(points, axis=0)
    return mu, points

def M2DP_desc(seg):
    """
        generate pointcloud signature
    """
    des, A1 = M2DP(seg)
    return des

def M2DP_iss_desc(seg):
    """
        generate pointcloud signature from 
        iss simplified pointcloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(seg)
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
    
    des, A1 = M2DP(keypoints.points)
    return des

def M2DP_downsample_desc(seg):
    """
        pointcloud signature downsampled to 
        VOXEL_SIZEd voxels
    """
    VOXEL_SIZE = 0.01
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(seg)
    downpcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    
    des, A1 = M2DP(downpcd.points)
    return des

def dist(x,y):
    '''
        ||euklidean distance ^2||
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    return np.linalg.norm((x-y)*(x-y)) 

def avg_desc(new, old, occurences):
    '''
        return avg of two signature vectors
        weighted according to how many times observed
    '''
    avg_des = (occurences*np.asarray(new) + np.asarray(old))/(occurences+1)
    return avg_des

stats = {'desc_time_total':0, 'desc_time': [],'reg_time_total': 0, 'movements':[]}

class DynObjectDB:
    def __init__(self, dyn_labels, params, descriptor=None):
        self.markerArray = MarkerArray()
        self.markerPublisher = rospy.Publisher('/dyn_objects', MarkerArray, queue_size=1)
        self.posePublisher = rospy.Publisher('/dyn_obj_poses', PoseArray, queue_size=1)
        #cannot find tf frame immediately after created listener -> init one here
        #https://stackoverflow.com/questions/54596517/ros-tf-transform-cannot-find-a-frame-which-actually-exists-can-be-traced-with-r
        self.tf_buffer = tf2_ros.Buffer() 
        #TransformListener object - Once the listener is created, it starts receiving tf2 transformations over the wire, and buffers them for up to 10 seconds
        self.listener = tf2_ros.TransformListener(self.tf_buffer)   
        #all segments including dynamic objects
        self.segments = {}
        self.segment_publisher = rospy.Publisher('static_object_segment', PointCloud2, queue_size=10)
        #viz segment_registration.cpp
        #self.register_segment = rospy.ServiceProxy('register_segment', obj_db_msgs.srv.RegisterSegment)
        self.stats = {'desc_time_total':0, 'desc_time': [],'reg_time_total': 0, 'movements':[]}

        self.REG_T = params['regT']  #threshold distance for pointcloud signatures - max nonsimilarity toleration
        self.MOVE_T = params['moveT']   #0.5 works fine on TUM
        self.M2DP_VOX_SIZE = params['m2dp_vox_size']
        self.FILTER_DYNAMIC = params['filter_dynamic'] 
        self.AVG_SIGN = params['average_signatures']
        self.DYN_CLASSES = dyn_labels
        self.WORLD_FRAME = "world"
        self.SIZE_K_MOVE_T = 0    #t = MOVE_T - SIZE_K_MOVE_T * size(pc) - increasing movement tolerance for larger pcds
        self.SPECIAL_TREAT_BG = False

        self.NEW_CAM_FRAME = None#"openni_camera"

        if descriptor != None:
            self.calc_descriptor = descriptor
        else:
            self.calc_descriptor = self.M2DP_downsample_desc
        if self.FILTER_DYNAMIC == False:
            print("Ignoring dynamic objects")

    def print_stats(self):
        if self.FILTER_DYNAMIC and len(self.segments) > 0:
            print("\n-----OBJECT DB-----")
            print("Params used:")
            print("REG_T =", self.REG_T)
            print("MOVE_T =", self.MOVE_T)
            if self.calc_descriptor == self.M2DP_downsample_desc:
                print("used M2DP descriptor with downsample VOX_SIZE =", self.M2DP_VOX_SIZE)
            else:
                print("used custom descriptor")
            for cat in self.segments:
                if cat in self.DYN_CLASSES:
                    print(cat,"-",coco_labels[cat], "- detected", len(self.segments[cat]), "DYNAMIC segments")
                else:
                    print(cat,"-", coco_labels[cat], "detected", len(self.segments[cat]), "segments")
            print("total registration time:", stats['reg_time_total'])
            print("total descriptors calculation time:", stats['desc_time_total'])
            print("desc. calculation time avg:", np.average(stats['desc_time']), 
                "max:", np.max(stats['desc_time']), "min:", np.min(stats['desc_time']))
            print("total movements:", len(stats['movements']))
            print("movement avg:", np.average(stats['movements']), 
                "max:", np.max(stats['movements']), "min:", np.min(stats['movements']))
            print("----------------------")
        else:
            print("...")

    def M2DP_downsample_desc(self, seg):
        """
            pointcloud signature downsampled to 
            VOXEL_SIZEd voxels
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg)
        downpcd = pcd.voxel_down_sample(voxel_size=self.M2DP_VOX_SIZE)
        
        des, A1 = M2DP(downpcd.points)
        return des
    
    def get_obj_marker(self, object):
        pub = rospy.Publisher('pose', MarkerArray, queue_size=1)
        marker = Marker()
        
        marker.header.frame_id = object["frame"]
        marker.id = object['id']    #id is only unique inside category - problems?
        marker.type = 2
        marker.action = 0        
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 200
        marker.color.a = 0.8
        marker.pose = object['pos'].pose

        return marker             
    
    #TODO: possibly faster searching by using signature as key in hashtable
    #algthough calc_descriptor is the bottleneck
    # poss. solutions: 
        # - upgrade to faster M2DP
        # - simplify poinclouds to keypoints
        # - implement in M2DP in c++? :( 
        # - limit db size
        # - keyframes -> might work!
        # - max time per descriptor
    #segment doesnt have one semantic label, but uses the most probable one? -- can change?
    def has_moved(self, obj, pc):
        '''
            detects pointcloud movement
            input is pointcloud centroid -> if occluded could be different
                --maybe if inside pointcloud not moving
            -> thresholding, t quite high
        '''
        r2 = [obj['pos'].pose.position.x, obj['pos'].pose.position.y, obj['pos'].pose.position.z]
        r1 = [obj['pos_prev'].pose.position.x, obj['pos_prev'].pose.position.y, obj['pos_prev'].pose.position.z]
        move = np.linalg.norm(np.asarray(r1) - np.asarray(r2))
        stats['movements'].append(move)
        print("object",obj["semantic_label"],obj["id"],'movement detection', move)     
        #consider pointcloud size for movement threshold
        if(move < self.MOVE_T - self.SIZE_K_MOVE_T * len(pc)): 
            return False

        return True 

    def register_segment(self, obj):
        '''
            registers segment into object ID using descriptor key
        '''
        cat = obj['semantic_label']
        if not cat in self.segments:
            self.segments[cat] = []

        if len(self.segments[cat]) == 0:
            self.segments[cat].append(obj)
        else: 
            dists = []  #compared distances
            argmin = 0  #closest segment
            dist_min = None
            #separate branch for BG segments
            #if default class compare to all segments ...maybe do it for all?
            if(cat == 0 and self.SPECIAL_TREAT_BG): 
                dist_min = 10000
                min_cat = 0  
                for c in self.segments: #loop through prototypes
                    for prot_idx in range(len(self.segments[c])):
                        prot = self.segments[c][prot_idx]
                        d = dist(prot["signature"], obj["signature"])
                        #stop if background segment too similar to dynamic one
                        #fixes when cnn fails?
                        if d < self.REG_T:
                            if d < dist_min:
                                dist_min = d
                                min_cat = c
                                argmin = prot_idx
                            #if any bg segment too similar to dynamic one, get rid of 
                            if ((min_cat != cat) and (min_cat in self.DYN_CLASSES)):
                                print("background paired with dyn obj")
                                return
                cat = min_cat   #object paired with different cat
                dists.append(dist_min)

            #if not BG segment
            #lookup segment in segment library only under its label category
            else:
                for prot in self.segments[cat]:
                    dists.append(dist(prot["signature"], obj["signature"]))
                argmin = np.argmin(dists)   #closest obj in its class    
                dist_min = dists[argmin]

            #print("min distance: ", dist_min)
            if(dist_min <= self.REG_T):    #paired with already seen 
                #seg already seen
                self.segments[cat][argmin]['observ_count'] += 1 #save poses for movement detect
                self.segments[cat][argmin]['pos_prev'] = self.segments[cat][argmin]['pos']
                self.segments[cat][argmin]['pos'] = obj['pos']
                #edit prototype signature
                if self.AVG_SIGN:
                    self.segments[cat][argmin]['signature'] = avg_desc(obj['signature'], 
                                                                        self.segments[cat][argmin]['signature'], 
                                                                        self.segments[cat][argmin]['observ_count'])
                obj['id'] = argmin
            else:
                #label previously not seen, add new to segment db
                obj['id'] = len(self.segments[cat]) #id inside its category list
                self.segments[cat].append(obj)
                print("new segment instance:", coco_labels[cat], obj['id'])
        
                # print("SEGMENTAS")
                # for c in self.segments:
                #     print(c, "-", coco_labels[c], len(self.segments[c]))

    def segment_callback(self, data):
            if not self.FILTER_DYNAMIC: 
                #just pass segments
                self.segment_publisher.publish(data)
            else:
                assert isinstance(data, PointCloud2)   
                seg = point_cloud2.read_points(data)    #generator object
                point = next(seg)   #get labels from the first point
                semantic_label = point[-1]
                instance_label = point[-2]
                
                static_segment = data
                if self.NEW_CAM_FRAME != None:
                    static_segment.header.frame_id = self.NEW_CAM_FRAME

                #calc descriptor            
                mu, pc = get_point_cloud(seg)  
                t1 = time.time()
                sign = self.calc_descriptor(np.asarray(pc))
                t2 = time.time()
                t_desc = t2 - t1
                
                #create object
                seg_frame = data.header.frame_id.replace('/','')
                pose = build_pose_msg(seg_frame, mu)    #build pose from segment frame and clouds centroid
                pos = self.tranform_pose(pose, seg_frame, self.WORLD_FRAME, time_stamp = data.header.stamp)
                obj = {"id": 0, "signature": sign,"instance_label":instance_label, "semantic_label":semantic_label, 
                    "observ_count": 0, 'pos':pos, 'pos_prev': 0, 'frame': seg_frame}
                
                #registration
                t1 = time.time()
                self.register_segment(obj)
                t2 = time.time()
                t_reg = t2 - t1
                print("testing obj", obj['semantic_label'], obj['id'], 
                "descriptor time:", t_desc, "registation time", t_reg)
                stats['desc_time_total'] += t_desc
                stats['desc_time'].append(t_desc)
                stats['reg_time_total'] += t_reg
                
                #movement detection
                #not dynamic class but could be moving
                if(obj['semantic_label'] not in self.DYN_CLASSES):
                    #if was seen
                    if(self.segments[obj['semantic_label']][obj['id']]['observ_count'] > 0):
                        if not self.has_moved(self.segments[semantic_label][obj['id']], pc):
                            #publish segment msg for mapping
                            print("PUBLISHING object- class", obj['semantic_label'], "id", obj['id'])
                            self.segment_publisher.publish(static_segment)
                        else:
                            print("MOVEMNT DETECTED", obj['id'], coco_labels[semantic_label])
                #dynamic -> dont publish, create marker
                else: 
                    #TODO: update markers only if position changed 
                    marker = self.get_obj_marker(obj)
                    self.segments[obj['semantic_label']][obj['id']]['marker'] = marker
                    #self.publish_object_poses()
                    self.publish_object_markers()

    def publish_object_markers(self):
        # build marker array and publish
        ma = MarkerArray()
        for dyn_class in self.DYN_CLASSES:
            if dyn_class in self.segments:
                for obj in self.segments[dyn_class]:
                    ma.markers.append(obj["marker"])
            self.markerPublisher.publish(ma)
        #print("Markers published")
    
    def publish_object_poses(self):
        # build pose array and publish
        p = PoseArray()
        for dyn_class in self.DYN_CLASSES:
            if dyn_class in self.segments:
                for obj in self.segments[dyn_class]:
                    p.poses.append(obj['pos'].pose)
            self.posePublisher.publish(p)
        print("Poses published")

    def tranform_pose(self, input_pose, from_frame, to_frame, time_stamp = rospy.Time()):
        #if !can_lookup_transform time_stamp = rospy.time() -- rospy.time() = most recent
        if not self.tf_buffer.can_transform(to_frame, from_frame, time_stamp):
            time_stamp = rospy.Time()
            print("using latest TF transform instead of timestamp match")

        try:
            trans = self.tf_buffer.lookup_transform(from_frame, to_frame, time_stamp)

        except(tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            raise 
        
        trans_pose = tf2_geometry_msgs.do_transform_pose(input_pose, trans)

        return trans_pose 

#TODO: zkontrolovat xyz zda neni prohazeno
def build_pose_msg(frame, position, orientation = [0,0,0,1]):    
    p = Pose()
    p.position.x = position[0]
    p.position.y = position[1]
    p.position.z = position[2]
    # Make sure the quaternion is valid and normalized
    p.orientation.x = orientation[0]
    p.orientation.y = orientation[1]
    p.orientation.z = orientation[2]
    p.orientation.w = orientation[3]

    pose_stamped = tf2_geometry_msgs.PoseStamped()
    pose_stamped.pose = p
    pose_stamped.header.frame_id = frame
    pose_stamped.header.stamp = rospy.Time.now()
    
    return pose_stamped


if __name__ == '__main__':
    rospy.init_node('object_db', disable_signals=True)  #disable signals?
    
    params = {
        "filter_dynamic": True,
        "regT":0.03,
        "moveT":0.35,
        "m2dp_vox_size": 0.02,
        "average_signatures": True
    } 
    #load ros params
    #rparam = rospy.get_param('gains')
    #print("loaded params:")
    #print(rparam)
    #pass dynamic objects list(COCO ids), params, (descriptor function)
    db = DynObjectDB([1], params)
    
    rospy.Subscriber('/depth_segmentation_node/object_segment', PointCloud2, db.segment_callback)
    rospy.on_shutdown(db.print_stats)
    rospy.spin()