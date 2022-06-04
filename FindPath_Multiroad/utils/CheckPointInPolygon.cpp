#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <bits/stdc++.h>
using namespace std;

bool PointInPolygon(pair<double, double> point, vector<vector<double>> polygon){
    // https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    // Check a point is in polygon or not 
    // ---------------------------------------------------------------//
    // -----------------------------TO DO-----------------------------//
    // ---------------------------------------------------------------//
    int nvert = polygon.size();
    int i, j, c = 0;
    for (i = 0, j = nvert-1; i < nvert; j = i++) {
        if ( ((polygon[i][1]>point.second) != (polygon[j][1]>point.second)) &&
        (point.first < (polygon[j][0]-polygon[i][0]) * (point.second-polygon[i][1]) / (polygon[j][1]-polygon[i][1]) + polygon[i][0]) )
        c = !c;
    }
    // ---------------------------------------------------------------//
    // ------------------------------END------------------------------//
    // ---------------------------------------------------------------//
    if(c == 1)
        return true;
    return false;
}

vector<bool> PointsInPolygon(vector<pair<double, double>> points, vector<vector<double>> polygon){
    // https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    vector<bool> res;
    // Check points is in polygon or not 
    // ---------------------------------------------------------------//
    // -----------------------------TO DO-----------------------------//
    // ---------------------------------------------------------------//
    int nvert = polygon.size();
    for(auto point: points) {
        int i, j, c = 0;
        for (i = 0, j = nvert-1; i < nvert; j = i++) {
            if ( ((polygon[i][1]>point.second) != (polygon[j][1]>point.second)) &&
            (point.first < (polygon[j][0]-polygon[i][0]) * (point.second-polygon[i][1]) / (polygon[j][1]-polygon[i][1]) + polygon[i][0]) )
            c = !c;
        }
        if(c == 1)
            res.push_back(true);
        else
            res.push_back(false);
    }
    // ---------------------------------------------------------------//
    // ------------------------------END------------------------------//
    // ---------------------------------------------------------------//
    return res;    
}


PYBIND11_MODULE(CheckPointInPolygonCplusplus, handle) {
    handle.doc() = "this code by Ha Hoang";
    handle.def("PointInPolygon", &PointInPolygon);
    handle.def("PointsInPolygon", &PointsInPolygon);
}