#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <bits/stdc++.h>
using namespace std;

const int UNBLOCKED = 0;
const int BLOCKED = -1;
const int TARGET = 1;
pair<int, int> convert_point_to_cell(double h_coordinate, double w_coordinate, int H, int W, int N, int M) {
    double delta_h = double(H)/double(N);
    double delta_w = double(W)/double(M);

    int x = ceil(h_coordinate / delta_h) - 1;
    int y = ceil(w_coordinate / delta_w) - 1 ;   
    return {x, y};
}
pair<double, double> convert_cell_to_point(int x, int y, int H, int W, int N, int M) {
    x++;
    y++;
    double delta_h = double(H)/double(N);
    double delta_w = double(W)/double(M);

    double h_coordinate = delta_h*(x + x - 1)/2;
    double w_coordinate = delta_w*(y + y - 1)/2;

    return {h_coordinate, w_coordinate};
}

vector<vector<int>> fill_road(vector<vector<vector<double>>> road, int H, int W, int N, int M, vector<vector<int>> grid){
    // https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
    if(grid.size() == 0) 
        grid = vector<vector<int>> (N, vector<int>(M, BLOCKED));
    for(auto sub_road: road) {
        for(int x = 0; x < N; x++) {
            for(int y = 0; y < M; y++) {
                pair<double, double> point = convert_cell_to_point(x, y, H, W, N, M);

                // Check point is in sub_road or not 
                // ---------------------------------------------------------------//
                // -----------------------------TO DO-----------------------------//
                // ---------------------------------------------------------------//
                int nvert = sub_road.size();
                int i, j, c = 0;
                for (i = 0, j = nvert-1; i < nvert; j = i++) {
                    if ( ((sub_road[i][1]>point.second) != (sub_road[j][1]>point.second)) &&
                    (point.first < (sub_road[j][0]-sub_road[i][0]) * (point.second-sub_road[i][1]) / (sub_road[j][1]-sub_road[i][1]) + sub_road[i][0]) )
                    c = !c;
                }
                // ---------------------------------------------------------------//
                // ------------------------------END------------------------------//
                // ---------------------------------------------------------------//
                if(c == 1) {
                    grid[x][y] = UNBLOCKED;
                }
            }
        }    
    }
    return grid;
}


PYBIND11_MODULE(FillRoadCplusplus, handle) {
    handle.doc() = "this code by Ha Hoang";
    handle.def("convert_point_to_cell", &convert_point_to_cell);
    handle.def("convert_cell_to_point", &convert_cell_to_point);
    handle.def("fill_road", &fill_road);
}