#include "graph.h"
class GraphCut {
private:
	Graph<double, double, double>* graph;
public:
	GraphCut();
	GraphCut(int, int);
	int addVertex();
	double maxFlow();
	void addVertexWeights(int, double, double);
	void addEdges(int, int, double);
	bool isSourceSegment(int);
};
