def Dijkstra(listOfNodes, neighbourhood, startNode):
    '''
    Static Dijkstra
    Attributes:
    listOfNodes: Lists Nodes to Search, FORMAT: ['A', 'B', 'C', 'D', ...]
    neighbourhood: List Distances From Node To Neighbours, FORMAT: {'A': {'B': 2, 'C': 4, 'D': 3}, 'B': {'A': 2, 'C': 3, 'D': 6}}
    startNode: Node where the vehicle is starting, FORMAT: 'A'
    '''

    unvisited = {node: float('inf') for node in listOfNodes} # Produce a dictionary with listOfNodes and set distance to infinity.
    print (unvisited)
    visited = {}
    current = startNode
    currentDistance = 0
    unvisited[current] = currentDistance
    
    while True:
        for neighbour, distance in neighbourhood[current].items():
            if neighbour not in unvisited: continue
            newDistance = currentDistance + distance
            if (unvisited[neighbour] == float('inf') or unvisited[neighbour] > newDistance):
                unvisited[neighbour] = newDistance
        visited[current] = currentDistance
        del unvisited[current]
        if not unvisited: break
        candidates = [node for node in unvisited.items() if node[1]]
        current, currentDistance = sorted(candidates, key = lambda x: x[1])[0]



    print (visited)


nodes = ('A', 'B', 'C', 'D', 'E', 'F', 'G')
distances = {
    'B': {'A': 5, 'D': 1, 'G': 2},
    'A': {'B': 5, 'D': 3, 'E': 12, 'F' :5},
    'D': {'B': 1, 'G': 1, 'E': 1, 'A': 3},
    'G': {'B': 2, 'D': 1, 'C': 2},
    'C': {'G': 2, 'E': 1, 'F': 16},
    'E': {'A': 12, 'D': 1, 'C': 1, 'F': 2},
    'F': {'A': 5, 'E': 2, 'C': 16}}

Dijkstra(nodes, distances, 'B')
