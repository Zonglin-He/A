#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

// Each physical return is counted once per distinct component along its nested path.
extern "C" void spread(int points, int levels, int nodes, int boxes, const double* xyz,
                       const int* paths, const double* query, const double* marginal,
                       double* probability, int* support) {
    std::array<std::vector<int>,3> order;
    for (int a=0;a<3;++a) {
        auto& ids=order[a];
        ids.resize(points);
        std::iota(ids.begin(),ids.end(),0);
        std::sort(ids.begin(),ids.end(),[&](int i,int j){return xyz[i*3+a]<xyz[j*3+a];});
    }
    for (int k=0;k<boxes;++k) {
        int axis=0, start=0, finish=points;
        double lower[3],upper[3];
        for (int a=0;a<3;++a) {
            const auto& ids=order[a];
            lower[a]=query[k*6+a]-.5*query[k*6+3+a];
            upper[a]=query[k*6+a]+.5*query[k*6+3+a];
            auto lo=std::lower_bound(ids.begin(),ids.end(),lower[a],[&](int i,double x){return xyz[i*3+a]<x;});
            auto hi=std::upper_bound(lo,ids.end(),upper[a],[&](double x,int i){return x<xyz[i*3+a];});
            if(a==0 || hi-lo<finish-start){axis=a;start=lo-ids.begin();finish=hi-ids.begin();}
        }
        double* row=probability+size_t(k)*nodes;
        int count=0;
        for(int t=start;t<finish;++t) {
            const int p=order[axis][t];bool inside=true;
            for(int a=0;a<3;++a) inside=inside && xyz[p*3+a]>=lower[a] && xyz[p*3+a]<=upper[a];
            if(!inside)continue;
            ++count;
            int previous=-1;
            for(int a=0;a<levels;++a){
                int v=paths[a*points+p];
                if(v>=0 && v!=previous) row[v]+=1.;
                previous=v;
            }
        }
        support[k]=count;
        if(count)for(int j=0;j<nodes;++j)row[j]*=marginal[j]/count;
    }
}

