#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <bits/stdc++.h>
using namespace std;

const int MAXN = 80;
const int MAXM = 80;


class BFS {
    private:
        const int INF = 100000000;
        const int SOURCE_LOWERBOUND = 2;
        const int MAX_SOURCEs = 20;
        const int TARGET = 1;
        const int BLOCKED = -1;
        const int UNBLOCKED = 0;
        #define NDIR 8
        #define NMAX 80
        #define MMAX 80

        int dx[NDIR] = {0, 0, -1, 1, 1, 1, -1, -1};
        int dy[NDIR] = {1, -1, 0, 0, -1, 1, -1, 1};

        int n, m;
        bool inq[NMAX][MMAX];
        int trace[NMAX][MMAX];
        vector<vector<int> >grid;
        vector<vector<int> >slots;

    public:
        BFS(int _n, int _m, vector<vector<int> > _grid, vector<vector<int> > _slots) : n(_n), m(_m){
            grid = _grid;
            slots = _slots;
        }
        inline void convert_1D_to_2D(int u, int &x, int &y, int n, int m) {
            x = u/m;
            y = u%m;
        }
        inline void convert_2D_to_1D(int &u, int x, int y, int n, int m) {
            u = x*m+y;
        }

        bool is_valid(int node_x, int node_y, int n, int m) {
            if(node_x < 0 || node_x >= n) return false;
            if(node_y < 0 || node_y >= m) return false;
            return true;
        }

        void lock(int x, int y) {
            for(auto slot: slots) {
                if((x >= slot[0]) && (x <= slot[2]) && (y >= slot[1]) && (y <= slot[3])) {
                    for(int i = slot[0]; i <= slot[2]; i++)
                        for(int j = slot[1]; j <= slot[3]; j++)
                            grid[i][j] = BLOCKED;
                    break;
                }
            }
        }
        vector<vector<int>> shortestPath() {
            vector<vector<int>> paths(0);

            int source_1D = -1;
            int sources[MAX_SOURCEs] = {-1};
            int nsource = 0;
            for(int i = 0; i < n; i++)
                for(int j = 0; j < m; j++) {
                    // inq[i][j] = false;
                    // trace[i][j] = -1;
                    if(grid[i][j] >= SOURCE_LOWERBOUND) {
                        convert_2D_to_1D(source_1D, i, j, n, m);
                        if(grid[i][j] - SOURCE_LOWERBOUND < MAX_SOURCEs) {
                            sources[grid[i][j]-SOURCE_LOWERBOUND] = source_1D;
                            nsource++;
                        }
                    }
                }
            queue<int> q;
            for(int isource = 0; isource < nsource; isource++) {
                source_1D = sources[isource];
                if(source_1D == -1) continue;
                memset(inq, false, sizeof(inq));
                memset(trace, -1, sizeof(trace));

                q.push(source_1D);
                int source_2D_x, source_2D_y;
                convert_1D_to_2D(source_1D, source_2D_x, source_2D_y, n, m);

                inq[source_2D_x][source_2D_y] = true;
                // printf("source#%d = %d %d\n", isource + SOURCE_LOWERBOUND, source_2D_x, source_2D_y);
                int oops_2D_x = -1;
                int oops_2D_y = -1;
                int oops_1D = -1;
                while(!q.empty()) {
                    bool stop = false;
                    int u_1D = q.front();
                    int u_2D_x, u_2D_y;
                    convert_1D_to_2D(u_1D, u_2D_x, u_2D_y, n, m);
                    q.pop();
                    int to_1D;
                    int to_2D_x, to_2D_y;
                    for(int i = 0; i < NDIR; i++) {
                        to_2D_x = u_2D_x + dx[i];
                        to_2D_y = u_2D_y + dy[i];
                        if(!is_valid(to_2D_x, to_2D_y, n, m))
                            continue;
                        if(inq[to_2D_x][to_2D_y])
                            continue;
                        if(grid[to_2D_x][to_2D_y] == TARGET) {
                            // printf("target = %d %d\n", to_2D_x, to_2D_y);
                            oops_2D_x = to_2D_x;
                            oops_2D_y = to_2D_y;
                            convert_2D_to_1D(oops_1D, oops_2D_x, oops_2D_y, n, m);
                            trace[to_2D_x][to_2D_y] = u_1D;
                            lock(oops_2D_x, oops_2D_y);  
                            stop = true;
                            break;
                        }
                        if(grid[to_2D_x][to_2D_y] != UNBLOCKED)
                            continue;
                        
                        convert_2D_to_1D(to_1D, to_2D_x, to_2D_y, n, m);
                        q.push(to_1D);
                        inq[to_2D_x][to_2D_y] = true;
                        trace[to_2D_x][to_2D_y] = u_1D;
                    }
                    if(stop)
                        break;
                }
                while(!q.empty())
                    q.pop();
                vector<int> path(0);
                while(oops_1D != -1) {
                    path.push_back(oops_1D);
                    convert_1D_to_2D(oops_1D, oops_2D_x, oops_2D_y, n, m);        
                    oops_1D = trace[oops_2D_x][oops_2D_y];
                }
                paths.push_back(path);
            }
            return paths;
        }
};

PYBIND11_MODULE(FindPathCplusplus, handle) {
    handle.doc() = "this code BFS by Ha Hoang";
    pybind11::class_<BFS>(handle, "BFS")
        .def(pybind11::init<int, int, vector<vector<int>>, vector<vector<int>> >())
        .def("shortestPath", &BFS::shortestPath)
        ;

}


// int main() {
//     freopen("1.inp", "r", stdin);
//     // Read grid from 1.inp
//     // ---------------------------------------------------------------//
//     // -----------------------------TO DO-----------------------------//
//     // ---------------------------------------------------------------//
//     int n, m;
//     cin >> n >> m;
//     printf("shape %dx%d\n", n, m);
    
//     vector<vector<int>> grid(n, vector<int>(m));
//     for(int i = 0; i < n; i++)
//         for(int j = 0; j < m; j++) 
//             cin >> grid[i][j];
//     // ---------------------------------------------------------------//
//     // ------------------------------END------------------------------//
//     // ---------------------------------------------------------------//

//     // Read slots from 1.inp
//     // ---------------------------------------------------------------//
//     // -----------------------------TO DO-----------------------------//
//     // ---------------------------------------------------------------//
//     int n_slots;
//     cin >> n_slots;
//     printf("number of slots %d\n", n_slots);

//     vector<vector<int> > slots(n_slots, vector<int>(4));
//     for(int i = 0; i < n_slots; i++)
//         for(int j = 0; j < 4; j++) 
//             cin >> slots[i][j];
//     // ---------------------------------------------------------------//
//     // ------------------------------END------------------------------//
//     // ---------------------------------------------------------------//


//     clock_t start, end;
//     start = clock();
//     // find the shortest Path
//     // ---------------------------------------------------------------//
//     // -----------------------------TO DO-----------------------------//
//     // ---------------------------------------------------------------//
//     BFS findpath(n, m, grid, slots);
//     vector<vector<int>> paths = findpath.shortestPath();
    
//     // ---------------------------------------------------------------//
//     // ------------------------------END------------------------------//
//     // ---------------------------------------------------------------//
//     end = clock();
//     double time = end - start;
//     printf("CPU time = %f seconds\n", time/CLOCKS_PER_SEC);
//     return 0;
// }
