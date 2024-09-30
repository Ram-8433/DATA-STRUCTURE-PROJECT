import tkinter as tk
from tkinter import messagebox
import heapq


# Node Class for Linked List
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


# Linked List Class
class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return f"Inserted: {data}"
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
            return f"Inserted: {data}"

    def delete(self, data):
        if not self.head:
            return "List is empty."
        if self.head.data == data:
            self.head = self.head.next
            return f"Deleted: {data}"
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return f"Deleted: {data}"
            current = current.next
        return f"{data} not found."

    def display(self):
        if not self.head:
            return "List is empty."
        current = self.head
        result = []
        while current:
            result.append(current.data)
            current = current.next
        return "Linked List: " + " -> ".join(map(str, result))


# Linked List GUI
class LinkedListGUI:
    def __init__(self, root):
        self.linked_list = LinkedList()
        self.root = root
        self.root.title("Linked List GUI")

        self.label = tk.Label(root, text="Enter value:")
        self.label.pack(pady=5)

        self.entry = tk.Entry(root, font=('Arial', 14))
        self.entry.pack(pady=5)

        self.create_button("Insert", self.insert)
        self.create_button("Delete", self.delete)
        self.create_button("Display", self.display)

        self.output = tk.Text(root, height=10, width=50, font=('Arial', 12))
        self.output.pack(pady=10)

    def create_button(self, text, command):
        button = tk.Button(self.root, text=text, command=command, font=('Arial', 14), width=12, height=2)
        button.pack(pady=5)

    def insert(self):
        value = self.entry.get()
        if value:
            result = self.linked_list.insert(value)
            self.entry.delete(0, tk.END)
            self.output.insert(tk.END, result + "\n")
        else:
            messagebox.showerror("Error", "Please enter a value")

    def delete(self):
        value = self.entry.get()
        if value:
            result = self.linked_list.delete(value)
            self.entry.delete(0, tk.END)
            self.output.insert(tk.END, result + "\n")
        else:
            messagebox.showerror("Error", "Please enter a value")

    def display(self):
        result = self.linked_list.display()
        self.output.insert(tk.END, result + "\n")


# Min Heap Class
class MinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, value):
        heapq.heappush(self.heap, value)
        return f"Inserted: {value}"

    def extract_min(self):
        if not self.heap:
            return "Heap is empty."
        return f"Extracted Min: {heapq.heappop(self.heap)}"

    def display(self):
        return "Heap: " + ", ".join(map(str, self.heap)) if self.heap else "Heap is empty."


# Min Heap GUI
class MinHeapGUI:
    def __init__(self, root):
        self.min_heap = MinHeap()
        self.root = root
        self.root.title("Min-Heap Priority Queue GUI")

        self.label = tk.Label(root, text="Enter value:")
        self.label.pack(pady=5)

        self.entry = tk.Entry(root, font=('Arial', 14))
        self.entry.pack(pady=5)

        self.create_button("Insert", self.insert)
        self.create_button("Extract Min", self.extract_min)
        self.create_button("Display", self.display)

        self.output = tk.Text(root, height=10, width=50, font=('Arial', 12))
        self.output.pack(pady=10)

    def create_button(self, text, command):
        button = tk.Button(self.root, text=text, command=command, font=('Arial', 14), width=12, height=2)
        button.pack(pady=5)

    def insert(self):
        value = self.entry.get()
        if value:
            result = self.min_heap.insert(int(value))
            self.entry.delete(0, tk.END)
            self.output.insert(tk.END, result + "\n")
        else:
            messagebox.showerror("Error", "Please enter a value")

    def extract_min(self):
        result = self.min_heap.extract_min()
        self.output.insert(tk.END, result + "\n")

    def display(self):
        result = self.min_heap.display()
        self.output.insert(tk.END, result + "\n")


# Stack Using Linked List Class
class StackLinkedList:
    def __init__(self):
        self.top = None

    def is_empty(self):
        return self.top is None

    def push(self, data):
        new_node = Node(data)
        new_node.next = self.top
        self.top = new_node
        return f"Pushed: {data}"

    def pop(self):
        if self.is_empty():
            return "Stack is empty."
        popped_value = self.top.data
        self.top = self.top.next
        return f"Popped: {popped_value}"

    def peek(self):
        if self.is_empty():
            return "Stack is empty."
        return f"Top Element: {self.top.data}"

    def display(self):
        if self.is_empty():
            return "Stack is empty."
        temp = self.top
        result = []
        while temp:
            result.append(str(temp.data))
            temp = temp.next
        return "Stack: " + " -> ".join(result)


# Stack Using Linked List GUI
class StackLinkedListGUI:
    def __init__(self, root):
        self.stack = StackLinkedList()
        self.root = root
        self.root.title("Stack Using Linked List GUI")

        self.label = tk.Label(root, text="Enter value:")
        self.label.pack(pady=5)

        self.entry = tk.Entry(root, font=('Arial', 14))
        self.entry.pack(pady=5)

        self.create_button("Push", self.push)
        self.create_button("Pop", self.pop)
        self.create_button("Peek", self.peek)
        self.create_button("Display", self.display)

        self.output = tk.Text(root, height=10, width=50, font=('Arial', 12))
        self.output.pack(pady=10)

    def create_button(self, text, command):
        button = tk.Button(self.root, text=text, command=command, font=('Arial', 14), width=12, height=2)
        button.pack(pady=5)

    def push(self):
        value = self.entry.get()
        if value:
            result = self.stack.push(value)
            self.entry.delete(0, tk.END)
            self.output.insert(tk.END, result + "\n")
        else:
            messagebox.showerror("Error", "Please enter a value")

    def pop(self):
        result = self.stack.pop()
        self.output.insert(tk.END, result + "\n")

    def peek(self):
        result = self.stack.peek()
        self.output.insert(tk.END, result + "\n")

    def display(self):
        result = self.stack.display()
        self.output.insert(tk.END, result + "\n")


# Queue Class
class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, item):
        self.items.append(item)
        return f"Enqueued: {item}"

    def dequeue(self):
        if self.is_empty():
            return "Queue is empty."
        return f"Dequeued: {self.items.pop(0)}"

    def is_empty(self):
        return len(self.items) == 0

    def display(self):
        return "Queue: " + ", ".join(map(str, self.items)) if self.items else "Queue is empty."


# Queue GUI
class QueueGUI:
    def __init__(self, root):
        self.queue = Queue()
        self.root = root
        self.root.title("Queue GUI")

        self.label = tk.Label(root, text="Enter value:")
        self.label.pack(pady=5)

        self.entry = tk.Entry(root, font=('Arial', 14))
        self.entry.pack(pady=5)

        self.create_button("Enqueue", self.enqueue)
        self.create_button("Dequeue", self.dequeue)
        self.create_button("Display", self.display)

        self.output = tk.Text(root, height=10, width=50, font=('Arial', 12))
        self.output.pack(pady=10)

    def create_button(self, text, command):
        button = tk.Button(self.root, text=text, command=command, font=('Arial', 14), width=12, height=2)
        button.pack(pady=5)

    def enqueue(self):
        value = self.entry.get()
        if value:
            result = self.queue.enqueue(value)
            self.entry.delete(0, tk.END)
            self.output.insert(tk.END, result + "\n")
        else:
            messagebox.showerror("Error", "Please enter a value")

    def dequeue(self):
        result = self.queue.dequeue()
        self.output.insert(tk.END, result + "\n")

    def display(self):
        result = self.queue.display()
        self.output.insert(tk.END, result + "\n")


# Binary Search Tree Node Class
class TreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key


# Binary Search Tree Class
class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = TreeNode(key)
            return f"Inserted: {key}"
        else:
            return self._insert_rec(self.root, key)

    def _insert_rec(self, node, key):
        if key < node.val:
            if node.left is None:
                node.left = TreeNode(key)
                return f"Inserted: {key}"
            else:
                return self._insert_rec(node.left, key)
        else:
            if node.right is None:
                node.right = TreeNode(key)
                return f"Inserted: {key}"
            else:
                return self._insert_rec(node.right, key)

    def display(self):
        return self._inorder(self.root)

    def _inorder(self, node):
        if node is None:
            return []
        return self._inorder(node.left) + [node.val] + self._inorder(node.right)

    def display_inorder(self):
        result = self.display()
        return "BST Inorder: " + ", ".join(map(str, result)) if result else "BST is empty."


# Binary Search Tree GUI
class BSTGUI:
    def __init__(self, root):
        self.bst = BST()
        self.root = root
        self.root.title("Binary Search Tree GUI")

        self.label = tk.Label(root, text="Enter value:")
        self.label.pack(pady=5)

        self.entry = tk.Entry(root, font=('Arial', 14))
        self.entry.pack(pady=5)

        self.create_button("Insert", self.insert)
        self.create_button("Display Inorder", self.display)

        self.output = tk.Text(root, height=10, width=50, font=('Arial', 12))
        self.output.pack(pady=10)

    def create_button(self, text, command):
        button = tk.Button(self.root, text=text, command=command, font=('Arial', 14), width=12, height=2)
        button.pack(pady=5)

    def insert(self):
        value = self.entry.get()
        if value:
            result = self.bst.insert(int(value))
            self.entry.delete(0, tk.END)
            self.output.insert(tk.END, result + "\n")
        else:
            messagebox.showerror("Error", "Please enter a value")

    def display(self):
        result = self.bst.display_inorder()
        self.output.insert(tk.END, result + "\n")


# Hash Table Class
class HashTable:
    def __init__(self):
        self.size = 10
        self.table = [[] for _ in range(self.size)]

    def hash_function(self, key):
        return hash(key) % self.size

    def insert(self, key, value):
        hash_index = self.hash_function(key)
        for i, (k, v) in enumerate(self.table[hash_index]):
            if k == key:
                self.table[hash_index][i] = (key, value)  # Update existing key
                return f"Updated: {key} -> {value}"
        self.table[hash_index].append((key, value))
        return f"Inserted: {key} -> {value}"

    def display(self):
        result = []
        for index, items in enumerate(self.table):
            result.append(f"{index}: " + ", ".join(f"{k}->{v}" for k, v in items))
        return "\n".join(result) if result else "Hash Table is empty."


# Hash Table GUI
class HashTableGUI:
    def __init__(self, root):
        self.hash_table = HashTable()
        self.root = root
        self.root.title("Hash Table GUI")

        self.label_key = tk.Label(root, text="Enter key:")
        self.label_key.pack(pady=5)

        self.entry_key = tk.Entry(root, font=('Arial', 14))
        self.entry_key.pack(pady=5)

        self.label_value = tk.Label(root, text="Enter value:")
        self.label_value.pack(pady=5)

        self.entry_value = tk.Entry(root, font=('Arial', 14))
        self.entry_value.pack(pady=5)

        self.create_button("Insert", self.insert)
        self.create_button("Display", self.display)

        self.output = tk.Text(root, height=10, width=50, font=('Arial', 12))
        self.output.pack(pady=10)

    def create_button(self, text, command):
        button = tk.Button(self.root, text=text, command=command, font=('Arial', 14), width=12, height=2)
        button.pack(pady=5)

    def insert(self):
        key = self.entry_key.get()
        value = self.entry_value.get()
        if key and value:
            result = self.hash_table.insert(key, value)
            self.entry_key.delete(0, tk.END)
            self.entry_value.delete(0, tk.END)
            self.output.insert(tk.END, result + "\n")
        else:
            messagebox.showerror("Error", "Please enter both key and value")

    def display(self):
        result = self.hash_table.display()
        self.output.insert(tk.END, result + "\n")


# Graph Class (Adjacency List)
class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
        return f"Added edge: {u} -> {v}"

    def display(self):
        result = []
        for node, edges in self.graph.items():
            result.append(f"{node}: " + ", ".join(map(str, edges)))
        return "\n".join(result) if result else "Graph is empty."


# Graph GUI
class GraphGUI:
    def __init__(self, root):
        self.graph = Graph()
        self.root = root
        self.root.title("Graph GUI")

        self.label_u = tk.Label(root, text="Enter source node:")
        self.label_u.pack(pady=5)

        self.entry_u = tk.Entry(root, font=('Arial', 14))
        self.entry_u.pack(pady=5)

        self.label_v = tk.Label(root, text="Enter destination node:")
        self.label_v.pack(pady=5)

        self.entry_v = tk.Entry(root, font=('Arial', 14))
        self.entry_v.pack(pady=5)

        self.create_button("Add Edge", self.add_edge)
        self.create_button("Display", self.display)

        self.output = tk.Text(root, height=10, width=50, font=('Arial', 12))
        self.output.pack(pady=10)

    def create_button(self, text, command):
        button = tk.Button(self.root, text=text, command=command, font=('Arial', 14), width=12, height=2)
        button.pack(pady=5)

    def add_edge(self):
        u = self.entry_u.get()
        v = self.entry_v.get()
        if u and v:
            result = self.graph.add_edge(u, v)
            self.entry_u.delete(0, tk.END)
            self.entry_v.delete(0, tk.END)
            self.output.insert(tk.END, result + "\n")
        else:
            messagebox.showerror("Error", "Please enter both source and destination nodes")

    def display(self):
        result = self.graph.display()
        self.output.insert(tk.END, result + "\n")


import tkinter as tk
from tkinter import simpledialog, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

class Graph:
    def __init__(self):
        self.graph = {}

    def add_vertex(self, vertex):
        """Add a vertex to the graph."""
        if vertex not in self.graph:
            self.graph[vertex] = {}
            return f"Vertex '{vertex}' added."
        else:
            return f"Vertex '{vertex}' already exists."

    def add_edge(self, vertex1, vertex2, direction, distance):
        """Add an edge between two vertices with direction and distance."""
        if vertex1 in self.graph and vertex2 in self.graph:
            self.graph[vertex1][vertex2] = {'direction': direction, 'distance': distance}
            self.graph[vertex2][vertex1] = {'direction': self._reverse_direction(direction), 'distance': distance}
            return f"Edge from '{vertex1}' to '{vertex2}' with direction '{direction}' and distance {distance} added."
        else:
            return "One or both vertices not found."

    def _reverse_direction(self, direction):
        """Reverse the direction for bidirectional edges."""
        direction_map = {'left': 'right', 'right': 'left', 'straight': 'straight'}
        return direction_map.get(direction, 'straight')

    def bfs(self, start_vertex):
        """Perform BFS traversal from a given start vertex."""
        if start_vertex not in self.graph:
            return f"Vertex '{start_vertex}' not found."
        
        visited = set()
        queue = deque([start_vertex])
        bfs_order = []
        highlighted_edges = []

        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                bfs_order.append(vertex)

                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
                        highlighted_edges.append((vertex, neighbor))
        self.visualize(highlighted_edges=highlighted_edges, highlighted_nodes=bfs_order)
        return " -> ".join(bfs_order)

    def dfs(self, start_vertex):
        """Perform DFS traversal from a given start vertex."""
        if start_vertex not in self.graph:
            return f"Vertex '{start_vertex}' not found."

        visited = set()
        stack = [start_vertex]
        dfs_order = []
        highlighted_edges = []

        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                dfs_order.append(vertex)

                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        stack.append(neighbor)
                        highlighted_edges.append((vertex, neighbor))
        self.visualize(highlighted_edges=highlighted_edges, highlighted_nodes=dfs_order)
        return " -> ".join(dfs_order)

    def visualize(self, highlighted_edges=None, highlighted_nodes=None):
        """Visualize the graph with optional highlighted nodes and edges."""
        G = nx.Graph()
        for vertex, edges in self.graph.items():
            for adjacent, attributes in edges.items():
                G.add_edge(vertex, adjacent, direction=attributes['direction'], weight=attributes['distance'])
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 10))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=3000, font_size=12, font_weight='bold')
        
        if highlighted_edges:
            nx.draw_networkx_edges(G, pos, edgelist=highlighted_edges, edge_color='red', width=2)
        if highlighted_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=highlighted_nodes, node_color='orange', node_size=3000)

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.4, font_size=10, font_color='blue')
        
        # Draw directions on edges
        for (u, v, data) in G.edges(data=True):
            direction = data['direction']
            mid_point = [(pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2]
            plt.text(mid_point[0], mid_point[1] + 0.03, direction, fontsize=10, ha='center', color='green')
        
        plt.title("Graph Visualization", fontsize=20)
        plt.show()
    
    def display(self):
        """Display the graph vertices and their edges."""
        if not self.graph:
            return "Graph is empty."
        
        graph_details = []
        for vertex, edges in self.graph.items():
            connections = ', '.join([f"{neighbor} (Distance: {attributes['distance']}, Direction: {attributes['direction']})" for neighbor, attributes in edges.items()])
            graph_details.append(f"{vertex}: {connections}")
        
        return "\n".join(graph_details)

class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Graph Visualization with Directions")
        self.graph = Graph()
        
        # Set a normal window size with minimum size
        self.root.geometry('1200x800')
        self.root.minsize(800, 600)
        
        # Create a main frame with padding
        self.main_frame = tk.Frame(root, bg='#e4e4e4')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create a text widget with larger font size
        self.output_text = tk.Text(self.main_frame, height=15, width=100, font=('Arial', 12), wrap=tk.WORD)
        self.output_text.pack(pady=20, padx=10, fill=tk.BOTH, expand=True)
        
        # Create a menu bar with a clean design
        self.menu = tk.Menu(root)
        root.config(menu=self.menu)
        self.graph_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Graph", menu=self.graph_menu)
        
        self.graph_menu.add_command(label="Add Vertex", command=self.add_vertex)
        self.graph_menu.add_command(label="Add Edge", command=self.add_edge)
        self.graph_menu.add_separator()
        self.graph_menu.add_command(label="Display Graph", command=self.display_graph)
        self.graph_menu.add_command(label="Visualize Graph", command=self.visualize_graph)
        self.graph_menu.add_separator()
        self.graph_menu.add_command(label="BFS", command=self.perform_bfs)
        self.graph_menu.add_command(label="DFS", command=self.perform_dfs)
        self.graph_menu.add_separator()
        
        # Add new option to display vertices and edges
        self.graph_menu.add_command(label="Show Vertices and Edges", command=self.display_vertices_edges)
        self.graph_menu.add_separator()
        self.graph_menu.add_command(label="Exit", command=root.quit)

    def add_vertex(self):
        try:
            vertex = simpledialog.askstring("Input", "Enter the vertex to add:")
            if vertex:
                result = self.graph.add_vertex(vertex)
                self.output_text.insert(tk.END, result + "\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error adding vertex: {e}")

    def add_edge(self):
        try:
            vertex1 = simpledialog.askstring("Input", "Enter the first vertex:")
            vertex2 = simpledialog.askstring("Input", "Enter the second vertex:")
            direction = simpledialog.askstring("Input", "Enter the direction (left/right/straight):")
            distance = simpledialog.askstring("Input", "Enter the distance between the vertices:")
            if vertex1 and vertex2 and direction and distance:
                try:
                    distance = float(distance)  # Convert distance to a float
                    result = self.graph.add_edge(vertex1, vertex2, direction, distance)
                    self.output_text.insert(tk.END, result + "\n")
                except ValueError:
                    messagebox.showerror("Invalid Input", "Please enter a valid number for the distance.")
        except Exception as e:
            messagebox.showerror("Error", f"Error adding edge: {e}")

    def display_graph(self):
        try:
            result = self.graph.display()
            self.output_text.insert(tk.END, "Graph:\n" + result + "\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying graph: {e}")

    def visualize_graph(self):
        try:
            self.graph.visualize()
        except Exception as e:
            messagebox.showerror("Error", f"Error visualizing graph: {e}")

    def perform_bfs(self):
        try:
            start_vertex = simpledialog.askstring("Input", "Enter the start vertex for BFS:")
            if start_vertex:
                result = self.graph.bfs(start_vertex)
                self.output_text.insert(tk.END, "BFS Traversal:\n" + result + "\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error performing BFS: {e}")

    def perform_dfs(self):
        try:
            start_vertex = simpledialog.askstring("Input", "Enter the start vertex for DFS:")
            if start_vertex:
                result = self.graph.dfs(start_vertex)
                self.output_text.insert(tk.END, "DFS Traversal:\n" + result + "\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error performing DFS: {e}")

    def display_vertices_edges(self):
        """Display vertices and edges of the graph in the output text widget."""
        try:
            result = self.graph.display()
            self.output_text.insert(tk.END, "Vertices and Edges:\n" + result + "\n")
        except Exception as e:
            messagebox.showerror("Error", f"Error displaying vertices and edges: {e}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = GraphApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


from PIL import Image, ImageTk

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Structures GUI")
        self.geometry("300x600")

        # Load the background image
        self.bg_image = Image.open("imagram1.jpg")  # Replace with your image path
        self.bg_image = self.bg_image.resize((2000, 1400), Image.LANCZOS)  # Resize image to fit the window
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)

        # Set the background image as a label
        self.bg_label = tk.Label(self, image=self.bg_photo)
        self.bg_label.place(relwidth=1, relheight=1)

        self.label = tk.Label(self, text="Select a Data Structure", font=("Helvetica", 16), bg='white')
        self.label.pack(pady=20)

        self.create_button("Stack (Linked List)", self.open_stack_gui)
        self.create_button("Linked List", self.open_linked_list_gui)
        self.create_button("Queue", self.open_queue_gui)
        self.create_button("Binary Search Tree", self.open_bst_gui)
        self.create_button("Hash Table", self.open_hash_table_gui)
        self.create_button("Graph", self.open_graph_gui)
        self.create_button("Min Heap", self.open_min_heap_gui)
        self.create_button("BFS AND DFS", self.open_Graph)
        self.create_button("Exit", self.quit)

    def create_button(self, text, command):
        button = tk.Button(self, text=text, command=command, font=('Arial', 14), width=20, height=2)
        button.pack(pady=10)

    def open_stack_gui(self):
        stack_window = tk.Toplevel(self)
        StackLinkedListGUI(stack_window)

    def open_linked_list_gui(self):
        linked_list_window = tk.Toplevel(self)
        LinkedListGUI(linked_list_window)

    def open_queue_gui(self):
        queue_window = tk.Toplevel(self)
        QueueGUI(queue_window)

    def open_bst_gui(self):
        bst_window = tk.Toplevel(self)
        BSTGUI(bst_window)

    def open_hash_table_gui(self):
        hash_table_window = tk.Toplevel(self)
        HashTableGUI(hash_table_window)

    def open_graph_gui(self):
        graph_window = tk.Toplevel(self)
        GraphGUI(graph_window)

    def open_min_heap_gui(self):
        min_heap_window = tk.Toplevel(self)
        MinHeapGUI(min_heap_window)

    def open_Graph(self):
        GraphApp_window = tk.Toplevel(self)
        GraphApp(GraphApp_window)

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
