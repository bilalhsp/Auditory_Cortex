import colorsys

class TreeNode:
    def __init__(self, label):
        self.label = label
        self.attributes = {'name': label}
        self.children = []

    def add_child(self, child):
        if isinstance(child, TreeNode):
            self.children.append(child)
        else:    
            self.children.append(TreeNode(child))
    
    def num_children(self):
        return len(self.children)
    
    def get_all_children(self):
        return self.children
    
    def set_hue(self, hue):
        self.attributes['hue'] = hue
        # print(f"Hue value set for {self.label}")
    

    def get_num_children(self):
        return len(self.children)
    
    def assign_hue(self, r, f, verbose=False):
        # print(f"Assigning hue to {self.label}")
        # print(f"{self.label} got the range: {r}")
        hue = (r[0]+r[1])/2
        self.set_hue(hue/360.0)
        n_children = self.get_num_children()
        # print(f"{n_children} children for node: {self.label}")
        if n_children > 0:
            reduced_r_list = []
            hue_range = r[1] - r[0]
            centers_gap = hue_range/n_children
            for i in range(n_children):

                r_start = r[0] + i*centers_gap
                r_end = r_start + centers_gap
                retained_gap = (r_end - r_start)*f
                skipped_gap = (r_end - r_start)*(1-f)
                # print(f"skipped gap: {skipped_gap}")
                strt = r_start + skipped_gap/2
                reduced_r_list.append((strt, strt + retained_gap))
                # print(reduced_r_list[i])

            for child, hue_range in zip(self.get_all_children(), reduced_r_list):
                child.assign_hue(hue_range, f)
        else:
            if verbose:
                print(f"At leaf node: {self.label}")


    def get_attribute(self, attr='name'):
        if attr in self.attributes.keys():
            return self.attributes[attr]
        return False
    
    def set_attribute(self, attr, value):
        self.attributes[attr] = value

    def is_leaf(self):
        return self.num_children() == 0
    
    def set_color_values(self):
        h = self.get_attribute('hue')
        l = self.get_attribute('luminance')
        s = self.get_attribute('chroma')
        
        if h and l and s:
            # print("Setting color values...")
            # h = h/360
            # l = l/100
            # s = s/100
            hls_color = (h,l,s)
            rgb_color = colorsys.hls_to_rgb(h,l,s)
            self.set_attribute('hls_color', hls_color)
            self.set_attribute('rgb_color', rgb_color)

        


class Tree:
    def __init__(self, edges, root, name='myTree'):
        self.nodes = {}
        self.root_label = root
        for edge in edges:
            self.add_edge(edge)

        self.update_tree_depths()

    # def linkage_to_edges(self, linkage):

    def get_node_attribute(self, label, attr):
        """Returns value of the attribute for mentioned node.
        """
        node = self.get_node(label)
        return node.get_attribute(attr)

    def add_node(self, label):
        if label not in self.nodes.keys():
            self.nodes[label] = TreeNode(label)
            if label == self.root_label:
                self.nodes[label].set_attribute('depth', 0)


    def get_node(self, label):
        if label in self.nodes.keys():
            return self.nodes[label]
        else:
            return -1
        
    def get_leaf_nodes(self):
        leaf_nodes = []
        for label, node in self.nodes.items():
            if node.is_leaf():
                leaf_nodes.append(label)

        return leaf_nodes
        

    def add_edge(self, label_tuple):
        for label in label_tuple:
            self.add_node(label)
        self.get_node(label_tuple[0]).add_child(
            self.get_node(label_tuple[1])
            )
        # set the node depth
        parent_depth = self.get_node(label_tuple[0]).get_attribute('depth')
        self.get_node(label_tuple[1]).set_attribute('depth', parent_depth+1)

    def parse_tree(self, label=None, attr='name'):
        if label is None:
            label = self.root_label    
        node = self.get_node(label)
        print(f"Node: {node.label}, {attr}: {node.get_attribute(attr)}")
        for child in node.get_all_children():
            self.parse_tree(child.label, attr=attr)

    def assign_hue(self, r, f, verbose=False):
        self.get_node(self.root_label).assign_hue(r,f, verbose=verbose)

    def assign_luminance(self, label=None, B_l=-10, L_1=70):
        if label is None:
            label = self.root_label
        node = self.get_node(label)
        if not node.is_leaf():
            for child in node.get_all_children():
                self.assign_luminance(child.label, B_l=B_l, L_1=L_1)
        depth_i = node.get_attribute('depth')
        luminance = (depth_i - 1)* B_l + L_1
        # valid values from 0-100 only.
        luminance = min(luminance, 100)
        luminance = max(luminance, 0)
        node.set_attribute('luminance', luminance/100.0)


    def assign_chroma(self, label=None, B_c=5, C_1=60):
        if label is None:
            label = self.root_label
        node = self.get_node(label)
        if not node.is_leaf():
            for child in node.get_all_children():
                self.assign_chroma(child.label, B_c=B_c, C_1=C_1)
        depth_i = node.get_attribute('depth')
        chroma = (depth_i - 1)* B_c + C_1
        # valid values from 0-100 only.
        chroma = min(chroma, 100)
        chroma = max(chroma, 0)

        node.set_attribute('chroma', chroma/100.0)


    def assign_HCL(self, r, f,
            B_l=-10, L_1=70, B_c=5, C_1=60,
            verbose=False):
        self.assign_hue(r, f, verbose=verbose)
        self.assign_chroma(B_c=B_c, C_1=C_1)
        self.assign_luminance(B_l=B_l, L_1=L_1)

        for label, node in self.nodes.items():
            node.set_color_values()



    def get_tree_depth(self):
        depth = 0
        for label, node in self.nodes.items():
            node_depth = node.get_attribute('depth')
            if node_depth > depth:
                depth = node_depth
        return depth
    
    def update_tree_depths(self, node=None, depth=0):
        """Makes sure the node depths are valid,
        need to run this at tree creation, because 
        depths of nodes for initial additions will
        depend upon the order in which edges are added.
        """
        if node is None:
            node = self.get_root_node()
            depth = 0
        if node != -1:
            node.set_attribute('depth', depth)
            for child in node.get_all_children():
                self.update_tree_depths(child, depth=depth+1)
        else:
            raise ValueError(f"Node {node} does not exist.'")

    def get_root_node(self):
        """Returns the root node of the tree"""
        return self.get_node(self.root_label)

    
    


    def get_nodes_at_depth(self, depth):
        """Returns ALL the nodes of the main tree that are
        at the specified depth. Always starts at the root node.

        Args:
            depth: int = level of tree we want to look at

        Returns:
            list = list of nodes labels having depth equal 
                to specified depth.
        """
        root = self.get_root_node()
        return self.get_subtree_nodes_at_depth(root, depth)
    
    def get_subtree_nodes_at_depth(self, node, depth):
        """Returns node labels from the sub-treeat the 
        specified depth. This method can be called on any
        subtree of the original tree (e.g. sub-tree
        starting at 'node')
        
        Args:
            node: Treenode = root of the subtree.
            depth: int = level of tree we want to look at

        Returns:
            list = list of nodes labels (can be the label
                of current node of labels of children
                in the subtree).
        """
        current_depth = node.get_attribute('depth')
        if current_depth == depth:
            return [node.label]
        elif current_depth > depth:
            return []
        else:
            node_labels = []
            children = node.get_all_children()
            for child in children:
                out = self.get_subtree_nodes_at_depth(child, depth)
                node_labels.extend(out)
            return node_labels


    def get_parent_node_label(self, label):
        """Returns label of the parent node for the given node label.
        
        Args:
            label = label of the any node of the tree

        Returns:
            Returns label of the immediate parent of input label.
        """
        for node_label, node in self.nodes.items():
            for child in node.get_all_children():
                if child.label == label:
                    return node_label











def linkage_to_edges(linkage_data):
    edges = []
    num_leaf_nodes = linkage_data.shape[0] + 1
    # print(num_leaf_nodes)

    max_leaf_label = num_leaf_nodes - 1
    root_label = max_leaf_label + linkage_data.shape[0]

    for i, arr in enumerate(linkage_data):
        parent = num_leaf_nodes + i 
        child1 = int(arr[0])
        child2 = int(arr[1])
        edges.append((parent, child1))
        edges.append((parent, child2))

    return edges, root_label

        