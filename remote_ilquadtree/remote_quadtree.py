import pickle
import ilquadtree
from collections import defaultdict
from bboxes import bbox_from_hierarchical_id
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
from bboxes import is_sub_bbox, point_is_inside_bbox, bboxes_intersect
import psutil

def get_percent_memory_used():
    total_mem = psutil.virtual_memory().total# /(2**20)
    used_mem = psutil.virtual_memory().used# /(2**20)
    return used_mem/total_mem

def get_node_depth(quadnode):
    depth = 0
    if quadnode.is_subdivided():
        depth = 1 + max([get_node_depth(c) for c in quadnode.children])
    return depth

def get_total_objects_recursive(quadnode):
    total_objects_recursive = quadnode.total_inner_objects
    if quadnode.is_subdivided():
        for child in quadnode.children:
            total_objects_recursive += get_total_objects_recursive(child)
    return total_objects_recursive


def replicate_structure(ram_quadtree, hierarchical_id, objects_remote_dir, parent_quadnode = None, output_children_position = None):
    bbox = ilquadtree.get_MBR(ram_quadtree)
    total_inner_objects = len(ram_quadtree.nodes)
    output_quadnode = RemoteQuadNode(bbox, total_inner_objects, hierarchical_id, objects_remote_dir)
    if parent_quadnode is not None:
        parent_quadnode.add_children_node(output_quadnode, output_children_position)
    if len(ram_quadtree.children) > 0:
        replicate_structure(ram_quadtree.children[0], hierarchical_id + '0', objects_remote_dir + '0', output_quadnode, 0)
        replicate_structure(ram_quadtree.children[1], hierarchical_id + '1', objects_remote_dir + '1', output_quadnode, 1)
        replicate_structure(ram_quadtree.children[2], hierarchical_id + '2', objects_remote_dir + '2', output_quadnode, 2)
        replicate_structure(ram_quadtree.children[3], hierarchical_id + '3', objects_remote_dir + '3', output_quadnode, 3)
    return output_quadnode

def get_descendent_nodes_info(quadnode, quadnode_code_id = '', nodes_info = None):
    if nodes_info is None:
        nodes_info = []
    nodes_info.append(
        {'node_object': quadnode, 
         'node_hierarchical_id': quadnode.hierarchical_id, 
         'bbox': quadnode.bbox, 
         'total_children_nodes': len(quadnode.children), 
         'total_inner_objects': quadnode.total_inner_objects}
    )
    child_nodes = quadnode.children.copy()
    if len(child_nodes)>0:
        for i, node in enumerate(child_nodes.copy()):
            get_descendent_nodes_info(node, quadnode_code_id = quadnode_code_id+str(i), nodes_info = nodes_info)
    return nodes_info


def create_objects_files(ram_quadtree, leaf_nodes_ids_and_bboxes, base_filename):
    for leaf_node_id, leaf_node_bbox in leaf_nodes_ids_and_bboxes:
        objects = []
        node = ram_quadtree
        for digit in leaf_node_id:
            candidate_objects = node.nodes
            for obj in candidate_objects:
                if point_is_inside_bbox(obj.item.centroid(), leaf_node_bbox):
                    objects.append(obj)
            node = node.children[int(digit)]
        objects.extend(node.nodes)
        objects = [o.item for o in objects]
        with open(f'{base_filename}{leaf_node_id}.pkl', 'wb') as f:
            pickle.dump(objects, f)

def obj_is_inside_bbox(obj, bbox):
    return point_is_inside_bbox(obj.centroid(), bbox)

class RemoteQuadNode:
    def __init__(self, bbox, total_inner_objects, hierarchical_id, objects_remote_dir = 'rootnode_'):
        self.children = []
        self.total_inner_objects = total_inner_objects
        self.objects_remote_dir = objects_remote_dir
        self.bbox = bbox
        xmin, ymin, xmax, ymax = bbox
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.center = ((xmin+xmax)/2, (ymin+ymax)/2)
        self.inner_objects = None
        self.hierarchical_id = hierarchical_id

    def add_children_node(self, node, position: int): # node is a RemoteQuadNode
        if len(self.children) == 0:
            self.children = [None,None,None,None]
        self.children[position] = node

    def is_subdivided(self):
        return len(self.children) != 0
    
    def get_descendent_nodes(self):
        descendent_nodes = self.children.copy()
        if len(descendent_nodes)>0:
            for node in descendent_nodes.copy():
                descendent_nodes.extend(node.get_descendent_nodes())
        return descendent_nodes
    
    def get_descendent_nodes_at_level(self, level: int):
        max_depth = get_node_depth(self)
        if level > max_depth:
            level = max_depth

        if level == 0:
            return [self]
        
        nodes = defaultdict(list)
        for l in range(1, level+1):
            # compute descendent nodes at level l
            nodes_previous_level = self.get_descendent_nodes_at_level(l-1)
            for node in nodes_previous_level:
                nodes[l].extend(node.children)
        return nodes[level]
    
    def get_descendent_nodes_up_to_level(self, level: int):
        max_depth = get_node_depth(self)
        if level > max_depth:
            level = max_depth

        if level == 0:
            return {0: [self]}
        
        nodes = defaultdict(list)
        for l in range(1, level+1):
            # compute descendent nodes at level l
            nodes_previous_level = self.get_descendent_nodes_up_to_level(l-1)[l-1]
            for node in nodes_previous_level:
                nodes[l].extend(node.children)
        return nodes
    
    def get_depth(self):
        return get_node_depth(self) # changes as the quadtree structure is being built
    

    def get_objects(self):
        if self.inner_objects is None:
            try:
                with open(self.objects_remote_dir + '.pkl', 'rb') as f:
                    self.inner_objects = pickle.load(f)
            except:
                #print('Not found:', self.objects_remote_dir + '.pkl')
                self.inner_objects = []
        return self.inner_objects
    
    def clean_objects_in_memory(self):
        del self.inner_objects
        self.inner_objects = None
    
    def get_inner_objects_recursive_costly(self):
        descendent_objects = self.get_objects()
        if self.is_subdivided():
            descendent_objects.extend(self.children[0].get_inner_objects_recursive_costly())
            descendent_objects.extend(self.children[1].get_inner_objects_recursive_costly())
            descendent_objects.extend(self.children[2].get_inner_objects_recursive_costly())
            descendent_objects.extend(self.children[3].get_inner_objects_recursive_costly())
        return descendent_objects
    
    def get_descendent_node_from_hierarchical_id(self, hierarchical_id: str):
        try:
            node = self
            for digit in hierarchical_id:
                node = node.children[int(digit)]
            return node
        except IndexError:
            print('Invalid code: path doesn\'t exist!')
            return None

    def _plot_geometries_recursive_costly(self, ax):
        for obj in self.get_objects():
            x1, y1, x2, y2 = obj.bbox()
            if x1==x2 and y1==y2:
                ax.scatter([x1,x2], [y1,y2], c='b')
            else:
                rect = Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='blue')
                ax.add_patch(rect)
            
        if self.is_subdivided():
            self.children[0]._plot_geometries_recursive_costly(ax)#self.children[0], bbox_from_hierarchical_id('0', self.bbox), ax)
            self.children[1]._plot_geometries_recursive_costly(ax)
            self.children[2]._plot_geometries_recursive_costly(ax)
            self.children[3]._plot_geometries_recursive_costly(ax)

    def _plot_bbox_recursive(self, ax):
        xmin, ymin, xmax, ymax = self.bbox
        ax.plot([xmin,xmax,xmax,xmin,xmin], [ymin,ymin,ymax,ymax,ymin], c='k')
        if self.is_subdivided():
            self.children[0]._plot_bbox_recursive(ax)#bbox_from_hierarchical_id('0', bbox), ax)
            self.children[1]._plot_bbox_recursive(ax)
            self.children[2]._plot_bbox_recursive(ax)
            self.children[3]._plot_bbox_recursive(ax)
    


class RemoteQuadtree:
    def __init__(self, ram_quadtree, objects_remote_dir = 'rootnode_'):
        self.root = replicate_structure(ram_quadtree, hierarchical_id = '', objects_remote_dir = objects_remote_dir)
        self.total_bbox = self.root.bbox
        self.objects_remote_dir = objects_remote_dir
        xmin, ymin, xmax, ymax = self.total_bbox
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.total_inner_objects = get_total_objects_recursive(self.root)
        self.structure = get_descendent_nodes_info(self.root)
        leaf_nodes_ids_and_bboxes = [(e['node_hierarchical_id'], e['bbox']) for e in self.structure if e['total_inner_objects']!=0]
        create_objects_files(ram_quadtree, leaf_nodes_ids_and_bboxes, base_filename = objects_remote_dir)
        self.depth = get_node_depth(self.root)

    def intersect(self, bbox):
        intersecting_objects = []
        candidate_intersecting_objects = []
        intersecting_nodes: list[RemoteQuadNode] = self.get_main_nodes_intersecting_bbox(bbox)
        for node in intersecting_nodes:
            if is_sub_bbox(node.bbox, bbox):
                intersecting_objects.extend(node.get_inner_objects_recursive_costly())
            else:
                candidate_intersecting_objects.extend(node.get_inner_objects_recursive_costly())

        #else:
        # for obj in candidate_intersecting_objects:
        #     if point_is_inside_bbox(obj.centroid(), bbox):
        #         intersecting_objects.append(obj)

        intersecting_objects.extend(list(filter(lambda obj: point_is_inside_bbox(obj.centroid(), bbox), candidate_intersecting_objects)))
  
        return intersecting_objects
    
    def get_main_nodes_intersecting_bbox(self, bbox):
        level = 0
        xmin, ymin, xmax, ymax = bbox
        width_bbox = xmax - xmin
        height_bbox = ymax - ymin
        while True:
            width_level, height_level = self.get_bbox_size_for_level(level)
            if (width_level < width_bbox and height_level < height_bbox) or (level >= self.depth):
                break
            level += 1
        main_intersecting_nodes = []
        nodes_up_to_level = self.root.get_descendent_nodes_up_to_level(level)
        for l in range(level+1):
            if l == level:
                for node in nodes_up_to_level[l]:
                    if bboxes_intersect(node.bbox, bbox):
                        main_intersecting_nodes.append(node)
            else:
                for node in nodes_up_to_level[l]:# for each node at level l
                    if (not node.is_subdivided()) and bboxes_intersect(node.bbox, bbox):
                        main_intersecting_nodes.append(node)
        return main_intersecting_nodes



        
    def get_bbox_size_for_level(self, level: int):
        width_level = self.width/(2**level)
        height_level = self.height/(2**level)
        return (width_level, height_level)

    def get_depth(self):
        return self.depth
    
    def get_total_bbox(self):
        return self.total_bbox
    
    def get_all_osm_ids_from_quadtree_costly(self):
        objs = self.get_objects_costly()
        osm_ids = [o.item['osm_id'] for o in objs]
        return osm_ids
    
    def get_total_objects(self):
        return self.total_inner_objects

    def get_objects_costly(self):
        all_objects = []
        descendent_nodes = [s['node_object'] for s in self.structure]
        for node in descendent_nodes:
            all_objects.extend(node.get_objects())
        return all_objects
    
    def balance_memory_allocation(self, acceptance_threshold = 0.90):
        percent_memory_alloc = get_percent_memory_used()
        if percent_memory_alloc > acceptance_threshold:
            self.clean_memory()

    def clean_memory(self):
        leaf_nodes = self.get_leaf_nodes()
        for node in leaf_nodes:
            node.clean_objects_in_memory()
    
    def get_nodes(self):
        return [s['node_object'] for s in self.structure]
    
    def get_leaf_nodes(self):
        leaf_nodes = [e['node_object'] for e in self.structure if e['total_inner_objects']!=0]
        return leaf_nodes


    def get_nodes_at_level(self, level: int):
        max_depth = self.get_depth()
        if level > max_depth:
            level = max_depth
        nodes_at_level = [s['node_object'] for s in self.structure if len(s['node_hierarchical_id']) == level]
        return nodes_at_level
    
    def get_node_from_hierarchical_id(self, hierarchical_id):
        return self.root.get_descendent_node_from_hierarchical_id(hierarchical_id)

        
    def plot(self, ax = None, include_objects_costly = False, hierarchical_ids_to_highlight = None):
        quadtree_direct_plot = (ax is None)
        if quadtree_direct_plot:
            _ = plt.figure(figsize = (10,7))
            ax = plt.subplot()
        if hierarchical_ids_to_highlight is None:
            hierarchical_ids_to_highlight = []
        bboxes_to_highlight = [bbox_from_hierarchical_id(hierarchical_id, self.bbox) for hierarchical_id in hierarchical_ids_to_highlight]

        self.root._plot_bbox_recursive(ax)
        if include_objects_costly:
            self.root._plot_geometries_recursive_costly(ax)

        # highlight the nodes with the selected hierarchical_ids:
        for bth in bboxes_to_highlight:
            x1, y1, x2, y2 = bth
            if x1==x2 and y1==y2:
                ax.scatter([x1,x2], [y1,y2], c='r')
            else:
                ax.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], c='r')

        if quadtree_direct_plot:
            xmin,ymin,xmax,ymax = self.total_bbox
            plt.xlim(xmin - 0.001, xmax + 0.001)
            plt.ylim(ymin - 0.001, ymax + 0.001)
            plt.show()
            


def plot_node_pairs(quadtree1, quadtree2, hierarchical_ids_pairs = None):
    if hierarchical_ids_pairs is None:
        hierarchical_ids_pairs = []
    total_bbox = quadtree1.total_bbox
    xmin,ymin,xmax,ymax = total_bbox
    x_plot_range = (xmin-0.001, xmax+0.001)
    y_plot_range = (ymin-0.001, ymax+0.001)
    fig, ax = plt.subplots(max(len(hierarchical_ids_pairs),1),2)
    fig.set_figwidth(7)
    fig.set_figheight(3*max(1,len(hierarchical_ids_pairs)))
    if len(hierarchical_ids_pairs) == 0:
        ax1 = ax[0]; ax1.set_xlim(*x_plot_range); ax1.set_ylim(*y_plot_range); ax1.set_xticks([]); ax1.set_yticks([])
        ax2 = ax[1]; ax2.set_xlim(*x_plot_range); ax2.set_ylim(*y_plot_range); ax2.set_xticks([]); ax2.set_yticks([])
        quadtree1.root._plot_bbox_recursive(ax1)
        quadtree2.root._plot_bbox_recursive(ax2)
    else:
        for i, ids_pair in enumerate(hierarchical_ids_pairs):
            ax1 = ax[i][0]; ax1.set_xlim(*x_plot_range); ax1.set_ylim(*y_plot_range); ax1.set_xticks([]); ax1.set_yticks([])
            ax2 = ax[i][1]; ax2.set_xlim(*x_plot_range); ax2.set_ylim(*y_plot_range); ax2.set_xticks([]); ax2.set_yticks([])
            quadtree1.root._plot_bbox_recursive(ax1)
            quadtree2.root._plot_bbox_recursive(ax2)
            bbox1 = bbox_from_hierarchical_id(ids_pair[0], total_bbox)
            bbox2 = bbox_from_hierarchical_id(ids_pair[1], total_bbox)

            x1, y1, x2, y2 = bbox1
            if x1==x2 and y1==y2:
                ax1.scatter([x1,x2], [y1,y2], c='r')
            else:
                ax1.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], c='r')
            

            x1, y1, x2, y2 = bbox2
            if x1==x2 and y1==y2:
                ax2.scatter([x1,x2], [y1,y2], c='r')
            else:
                ax2.plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], c='r')

    plt.show()