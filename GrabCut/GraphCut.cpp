#include "GraphCut.h"
GraphCut::GraphCut() {}
GraphCut::GraphCut(int _vCount, int _eCount) {
	graph = new Graph<double, double, double>(_vCount, _eCount);
}
int GraphCut::addVertex() {
	return graph->add_node();
}
double GraphCut::maxFlow() {
	return graph->maxflow();
}
void GraphCut::addVertexWeights(int _vNum, double _sourceWeight, double _sinkWeight) {
	graph->add_tweights(_vNum, _sourceWeight, _sinkWeight);
}
void GraphCut::addEdges(int _vNum1, int _vNum2, double _weight) {
	graph->add_edge(_vNum1, _vNum2, _weight, _weight);
}
bool GraphCut::isSourceSegment(int _vNum) {
	if (graph->what_segment(_vNum) == Graph<double, double, double>::SOURCE)return true;
	else return false;
}