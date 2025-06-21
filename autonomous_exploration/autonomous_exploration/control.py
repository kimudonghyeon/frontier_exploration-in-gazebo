import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid , Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import heapq , math , random , yaml
import scipy.interpolate as si
import sys , threading , time


with open("turtlebot3_ws/src/autonomous_exploration/config/params.yaml", 'r') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

lookahead_distance = params["lookahead_distance"]
speed = params["speed"]
expansion_size = params["expansion_size"]
target_error = params["target_error"]
robot_r = params["robot_r"]

pathGlobal = 0

def euler_from_quaternion(x,y,z,w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    return yaw_z

def heuristic(a, b):
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def astar(array, start, goal):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []
    heapq.heappush(oheap, (fscore[start], start))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data = data + [start]
            data = data[::-1]
            return data
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue
            if  tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1]for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    # If no path to goal was found, return closest path to goal
    if goal not in came_from:
        closest_node = None
        closest_dist = float('inf')
        for node in close_set:
            dist = heuristic(node, goal)
            if dist < closest_dist:
                closest_node = node
                closest_dist = dist
        if closest_node is not None:
            data = []
            while closest_node in came_from:
                data.append(closest_node)
                closest_node = came_from[closest_node]
            data = data + [start]
            data = data[::-1]
            return data
    return False

def bspline_planning(array, sn):
    try:
        array = np.array(array)
        x = array[:, 0]
        y = array[:, 1]
        N = 2
        t = range(len(x))
        x_tup = si.splrep(t, x, k=N)
        y_tup = si.splrep(t, y, k=N)

        x_list = list(x_tup)
        xl = x.tolist()
        x_list[1] = xl + [0.0, 0.0, 0.0, 0.0]

        y_list = list(y_tup)
        yl = y.tolist()
        y_list[1] = yl + [0.0, 0.0, 0.0, 0.0]

        ipl_t = np.linspace(0.0, len(x) - 1, sn)
        rx = si.splev(ipl_t, x_list)
        ry = si.splev(ipl_t, y_list)
        path = [(rx[i],ry[i]) for i in range(len(rx))]
    except:
        path = array
    return path

def pure_pursuit(current_x, current_y, current_heading, path, index):
    global lookahead_distance
    closest_point = None
    v = max(0.05, speed*0.6)

    # 경로 끝까지 보면서 lookahead 거리 이상 떨어진 점 찾기
    for i in range(index, len(path)):
        x = path[i][0]
        y = path[i][1]
        distance = math.hypot(current_x - x, current_y - y)
        if distance >= lookahead_distance:
            closest_point = (x, y)
            index = i
            break

    # 못 찾으면 마지막 점을 사용
    if closest_point is None:	
        closest_point = path[-1]
        index = len(path) - 1

    # 헤딩 계산
    target_heading = math.atan2(closest_point[1] - current_y, closest_point[0] - current_x)
    desired_steering_angle = target_heading - current_heading

    # 헤딩 wrap-around 처리
    if desired_steering_angle > math.pi:
        desired_steering_angle -= 2 * math.pi
    elif desired_steering_angle < -math.pi:
        desired_steering_angle += 2 * math.pi

    # steering angle이 너무 크면 제자리 회전만 → 너무 멈추는 경우 방지
    max_steer = math.pi / 4  # 더 유연하게 설정 가능
    if abs(desired_steering_angle) > max_steer:
        sign = 1 if desired_steering_angle > 0 else -1
        desired_steering_angle = sign * max_steer
        v = 0.05  # 조금만 이동

    return v, desired_steering_angle, index

def frontierB(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 0.0:
                if i > 0 and matrix[i-1][j] < 0:
                    matrix[i][j] = 2
                elif i < len(matrix)-1 and matrix[i+1][j] < 0:
                    matrix[i][j] = 2
                elif j > 0 and matrix[i][j-1] < 0:
                    matrix[i][j] = 2
                elif j < len(matrix[i])-1 and matrix[i][j+1] < 0:
                    matrix[i][j] = 2
    return matrix


def assign_groups(matrix):
    group = 1
    groups = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 2:
                group = dfs(matrix, i, j, group, groups)
    return matrix, groups

def dfs(matrix, i, j, group, groups):
    if i < 0 or i >= len(matrix) or j < 0 or j >= len(matrix[0]):
        return group
    if matrix[i][j] != 2:
        return group
    if group in groups:
        groups[group].append((i, j))
    else:
        groups[group] = [(i, j)]
    matrix[i][j] = 0
    dfs(matrix, i + 1, j, group, groups)
    dfs(matrix, i - 1, j, group, groups)
    dfs(matrix, i, j + 1, group, groups)
    dfs(matrix, i, j - 1, group, groups)
    dfs(matrix, i + 1, j + 1, group, groups) # sağ alt çapraz
    dfs(matrix, i - 1, j - 1, group, groups) # sol üst çapraz
    dfs(matrix, i - 1, j + 1, group, groups) # sağ üst çapraz
    dfs(matrix, i + 1, j - 1, group, groups) # sol alt çapraz
    return group + 1

from sklearn.cluster import KMeans

def find_frontiers_kmeans(matrix):
    frontiers = []
    for i in range(1, len(matrix) - 1):
        for j in range(1, len(matrix[0]) - 1):
            if matrix[i][j] == 0.0:
                if (matrix[i+1][j] < 0 or matrix[i-1][j] < 0 or
                    matrix[i][j+1] < 0 or matrix[i][j-1] < 0):
                    frontiers.append((i, j))
    return frontiers

def cluster_frontiers_kmeans(frontiers, k):
    if len(frontiers) < k:
        return []
    kmeans = KMeans(n_clusters=k, random_state=0).fit(frontiers)
    centroids = kmeans.cluster_centers_
    return [tuple(map(int, c)) for c in centroids]

def find_closest_kmeans_target(matrix, centroids, current, resolution, originX, originY):
    best_path = None
    best_dist = float('inf')
    fallback_path = None
    for c in centroids:
        path = astar(matrix, current, (int(c[0]), int(c[1])))
        if not path:
            continue
        world_path = [(p[1]*resolution+originX, p[0]*resolution+originY) for p in path]
        d = pathLength(world_path)
        if d > 0.3:
            best_dist = d
            best_path = world_path
        elif not fallback_path and d > target_error:
            fallback_path = world_path
    if best_path:
        return best_path
    elif fallback_path:
        print("[DEBUG] fallback path 사용")
        return fallback_path
    return None

def fGroups(groups):
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    top_five_groups = [(gid, cells) for gid, cells in sorted_groups[:5] if len(cells) > 2]
    return top_five_groups

def calculate_centroid(x_coords, y_coords):
    n = len(x_coords)
    sum_x = sum(x_coords)
    sum_y = sum(y_coords)
    mean_x = sum_x / n
    mean_y = sum_y / n
    centroid = (int(mean_x), int(mean_y))
    return centroid

#Bu fonksiyon en buyuk 5 gruptan target_error*2 uzaklıktan daha uzak olan ve robota en yakın olanı seçer.
"""
def findClosestGroup(matrix,groups, current,resolution,originX,originY):
    targetP = None
    distances = []
    paths = []
    min_index = -1
    for i in range(len(groups)):
        middle = calculate_centroid([p[0] for p in groups[i][1]],[p[1] for p in groups[i][1]]) 
        path = astar(matrix, current, middle)
        path = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
        total_distance = pathLength(path)
        distances.append(total_distance)
        paths.append(path)
    for i in range(len(distances)):
        if distances[i] > target_error*3:
            if min_index == -1 or distances[i] < distances[min_index]:
                min_index = i
    if min_index != -1:
        targetP = paths[min_index]
    else: #gruplar target_error*2 uzaklıktan daha yakınsa random bir noktayı hedef olarak seçer. Bu robotun bazı durumlardan kurtulmasını sağlar.
        index = random.randint(0,len(groups)-1)
        target = groups[index][1]
        target = target[random.randint(0,len(target)-1)]
        path = astar(matrix, current, target)
        targetP = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
    return targetP
"""
def findClosestGroup(matrix,groups, current,resolution,originX,originY):
    targetP = None
    distances = []
    paths = []
    score = []
    max_score = -1 #max score index
    for i in range(len(groups)):
        middle = calculate_centroid([p[0] for p in groups[i][1]],[p[1] for p in groups[i][1]]) 
        path = astar(matrix, current, middle)
        path = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
        total_distance = pathLength(path)
        distances.append(total_distance)
        paths.append(path)
    for i in range(len(distances)):
        if distances[i] <= 0.001:
            score.append(0)
        else:
            score.append(len(groups[i][1])/distances[i])
    for i in range(len(distances)):
        if distances[i] > target_error*3:
            print(f"[DEBUG] distances[i] > target_error*3")
            if max_score == -1 or score[i] > score[max_score]:
                max_score = i
    if max_score != -1:
        targetP = paths[max_score]
    else: #gruplar target_error*2 uzaklıktan daha yakınsa random bir noktayı hedef olarak seçer. Bu robotun bazı durumlardan kurtulmasını sağlar.
        index = random.randint(0,len(groups)-1)
        target = groups[index][1]
        target = target[random.randint(0,len(target)-1)]
        path = astar(matrix, current, target)
        targetP = [(p[1]*resolution+originX,p[0]*resolution+originY) for p in path]
    return targetP

def pathLength(path):
    for i in range(len(path)):
        path[i] = (path[i][0],path[i][1])
        points = np.array(path)
    differences = np.diff(points, axis=0)
    distances = np.hypot(differences[:,0], differences[:,1])
    total_distance = np.sum(distances)
    return total_distance

def costmap(data,width,height,resolution):
    data = np.array(data).reshape(height,width)
    wall = np.where(data == 100)
    for i in range(-expansion_size,expansion_size+1):
        for j in range(-expansion_size,expansion_size+1):
            if i  == 0 and j == 0:
                continue
            x = wall[0]+i
            y = wall[1]+j
            x = np.clip(x,0,height-1)
            y = np.clip(y,0,width-1)
            data[x,y] = 100
    data = data*resolution
    return data

def exploration(data, width, height, resolution, column, row, originX, originY, k=5):
    global pathGlobal
    data = costmap(data, width, height, resolution)
    data[row][column] = 0
    data[data > 5] = 1
    frontiers = find_frontiers_kmeans(data)
    if len(frontiers) < 5:
        print("[DEBUG] frontier 너무 적음 → 탐색 실패")
        pathGlobal = -1
        return
    centroids = cluster_frontiers_kmeans(frontiers, k)
    if len(centroids) == 0:
        print("[DEBUG] 클러스터링 실패")
        pathGlobal = -1
        return
    #data[data < 0] = 1
    path = find_closest_kmeans_target(data, centroids, (row, column), resolution, originX, originY)
    if not path or pathLength(path) < 0.3:
        print("[DEBUG] 유효한 경로 없음 또는 너무 짧음")
        pathGlobal = -1
        return
    if path:
        path = bspline_planning(path, len(path)*5)
    else:
        path = -1
    pathGlobal = path
    return

def localControl(scan):
    v = None
    w = None
    for i in range(60): #60
        if scan[i] < robot_r:
            v = 0.2
            w = -math.pi/4 
            break
    if v == None:
        for i in range(300,360): #300,360
            if scan[i] < robot_r:
                v = 0.2
                w = math.pi/4
                break
    return v,w

class navigationControl(Node):
    def __init__(self):
        super().__init__('Exploration')
        self.subscription = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.subscription = self.create_subscription(Odometry, 'odom', self.odom_callback, 10)
        self.subscription = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        print("탐색 모드 활성화")
        self.kesif = True
        self.start_time = time.time()  # 탐색 모드 시작 시간 기록
        threading.Thread(target=self.exp).start()  # 탐색 함수는 별도의 스레드로 실행
        
    def exp(self):
        twist = Twist()
        while True:  # 센서 데이터가 오기를 기다립니다.
            if not hasattr(self, 'map_data') or not hasattr(self, 'odom_data') or not hasattr(self, 'scan_data'):
                time.sleep(0.1)
                continue
            if self.kesif == True:
                if isinstance(pathGlobal, int) and pathGlobal == 0:
                    column = int((self.x - self.originX) / self.resolution)
                    row = int((self.y - self.originY) / self.resolution)
                    exploration(self.data, self.width, self.height, self.resolution, column, row, self.originX, self.originY)
                    self.path = pathGlobal
                else:
                    self.path = pathGlobal
                if isinstance(self.path, int) and self.path == -1:
                    #print("탐색 완료")
                    #end_time = time.time()  # 탐색 완료 시간 기록
                    #elapsed_time = end_time - self.start_time  # 시작 시간과 끝 시간 차이 계산
                    #minutes = int(elapsed_time // 60)  # 분 단위로 변환
                    #seconds = int(elapsed_time % 60)  # 초 단위로 변환
                    #print(f"탐색 완료, 소요 시간: {minutes}분 {seconds}초")
                    #sys.exit()
                    column = int((self.x - self.originX) / self.resolution)
                    row = int((self.y - self.originY) / self.resolution)
                    exploration(self.data, self.width, self.height, self.resolution, column, row, self.originX, self.originY)
                    self.path = pathGlobal
                    if isinstance(self.path, int) and self.path == -1:
                        print("[DEBUG] 재시도 실패 → 대기")
                        time.sleep(2.0)
                        self.kesif = True
                        continue
                else:
                    self.c = int((self.path[-1][0] - self.originX) / self.resolution)
                    self.r = int((self.path[-1][1] - self.originY) / self.resolution)
                    self.kesif = False
                    self.i = 0
                    print("새로운 목표 설정")
                    t = pathLength(self.path) / speed
                    t = t - 0.2  # x = v * t 공식에 따라 시간에서 0.2초를 뺌
                    self.t = threading.Timer(t, self.target_callback)  # 목표 도달 전 0.2초 남았을 때 탐색 함수 실행
                    self.t.start()
            
            # Rota 추적 블록 시작
            else:
                v, w = localControl(self.scan)
                if v == None:
                    v, w, self.i = pure_pursuit(self.x, self.y, self.yaw, self.path, self.i)
                if abs(self.x - self.path[-1][0]) < target_error and abs(self.y - self.path[-1][1]) < target_error:
                    v = 0.0
                    w = 0.0
                    self.kesif = True
                    print("목표 도달")
                    self.t.join()  # 스레드가 끝날 때까지 대기
                twist.linear.x = v
                twist.angular.z = w
                self.publisher.publish(twist)
                time.sleep(0.1)
            # Rota 추적 블록 종료

    def target_callback(self):
        exploration(self.data, self.width, self.height, self.resolution, self.c, self.r, self.originX, self.originY)
        
    def scan_callback(self, msg):
        self.scan_data = msg
        self.scan = msg.ranges

    def map_callback(self, msg):
        self.map_data = msg
        self.resolution = self.map_data.info.resolution
        self.originX = self.map_data.info.origin.position.x
        self.originY = self.map_data.info.origin.position.y
        self.width = self.map_data.info.width
        self.height = self.map_data.info.height
        self.data = self.map_data.data

    def odom_callback(self, msg):
        self.odom_data = msg
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = euler_from_quaternion(msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                         msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

def main(args=None):
    rclpy.init(args=args)
    navigation_control = navigationControl()
    rclpy.spin(navigation_control)
    navigation_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

