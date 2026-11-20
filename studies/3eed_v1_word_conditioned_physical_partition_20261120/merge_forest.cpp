// Maximal physically connected supports fitting each frozen noun's metric envelope.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {
constexpr int axes = 16, words = 7;
using Spans = std::array<double, axes+1>;
}

extern "C" int maximal_cuts(int n, int edges, const double *xyz, const int *left,
                            const int *right, const double *cost, const double *directions,
                            const double *dims, int *out_parent, int *out_size,
                            uint8_t *out_marks, int *out_assignment, int *out_coverage) {
    if (!n) return 0;
    const int capacity = 2*n;
    std::vector<int> leader(n), root(n), count(capacity,1), parent(capacity,-1);
    std::vector<double> height(capacity,-1.);
    std::vector<Spans> low(capacity), high(capacity);
    std::iota(leader.begin(),leader.end(),0);
    std::iota(root.begin(),root.end(),0);
    for (int i=0;i<n;++i) {
        for (int a=0;a<axes;++a)
            low[i][a] = xyz[3*i]*directions[2*a] + xyz[3*i+1]*directions[2*a+1];
        low[i][axes] = xyz[3*i+2];
        high[i]=low[i];
    }
    auto find = [&](int v) {
        while (v!=leader[v]) {leader[v]=leader[leader[v]];v=leader[v];}
        return v;
    };
    int total=n;
    for (int e=0;e<edges;++e) {
        int u=find(left[e]), v=find(right[e]);
        if (u==v) continue;
        if (count[root[u]]<count[root[v]]) std::swap(u,v);
        int a=root[u], b=root[v], k=total++;
        parent[a]=parent[b]=k;
        count[k]=count[a]+count[b]; height[k]=cost[e];
        for (int j=0;j<=axes;++j) {
            low[k][j]=std::min(low[a][j],low[b][j]);
            high[k][j]=std::max(high[a][j],high[b][j]);
        }
        leader[v]=u;root[u]=k;
    }
    std::vector<int> plateau(total), fit(total,0), marks(total,0), packed(total,-1);
    for (int v=total-1;v>=0;--v) {
        int p=parent[v];
        // A multiway merge at one radius has no intermediate physical supports.
        plateau[v]=(p>=0 && height[p]==height[v])?plateau[p]:v;
        if (plateau[v]!=v) continue;
        Spans span;
        for (int j=0;j<=axes;++j) span[j]=high[v][j]-low[v][j];
        for (int c=0;c<words;++c) {
            const double *d=dims+3*c;
            if (!(d[0]>0 && d[1]>0 && d[2]>0 && span[axes]<=d[2])) continue;
            bool possible=false;
            for (int j=0;j<axes && !possible;++j)
                possible=(span[j]<=d[0] && span[(j+axes/2)%axes]<=d[1]);
            if (possible) fit[v]|=1<<c;
        }
        marks[v]=fit[v] & ~(p<0?0:fit[plateau[p]]);
        for (int c=0;c<words;++c)
            if (marks[v]&(1<<c)) out_coverage[c]+=count[v];
    }
    int used=0;
    for (int v=0;v<total;++v) if (marks[v] && count[v]>=8) {
        int k=used++; packed[v]=k;out_size[k]=count[v];
        for (int c=0;c<words;++c) out_marks[words*k+c]=(marks[v]>>c)&1;
    }
    std::vector<int> ancestor(total,-1);
    std::vector<std::array<int,words>> label(total);
    for (int v=total-1;v>=0;--v) {
        int p=parent[v], k=packed[v];
        if (p<0) label[v].fill(-1); else label[v]=label[p];
        int next=p<0?-1:ancestor[p];
        ancestor[v]=(k<0)?next:k;
        if (k>=0) {
            out_parent[k]=next;
            for (int c=0;c<words;++c)
                if (marks[v]&(1<<c)) label[v][c]=k;
        }
        if (v<n) for (int c=0;c<words;++c) out_assignment[c*n+v]=label[v][c];
    }
    return used;
}
