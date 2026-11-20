// Exact probabilities of named endpoints and absence of intervening objects on a return tree.
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

struct Forest {
    int n;
    const int *parent;
    std::vector<std::vector<int>> child;
    std::vector<int> roots;
    std::vector<double> emit, split, message;
    Forest(int count,const int *p):n(count),parent(p),child(n),emit(n),split(n),message(n) {
        for(int i=0;i<n;++i) {
            if(p[i]>=0) child[p[i]].push_back(i);
            else roots.push_back(i);
        }
    }
    void recalc(int i) {
        double sub=split[i];
        for(int v: child[i]) sub*=message[v];
        message[i]=emit[i]+sub;
    }
    double total() const {
        double result=1.;
        for(int v:roots)result*=message[v];
        return result;
    }
    double change(int node,double value) {
        emit[node]=value;
        for(int v=node;v>=0;v=parent[v])recalc(v);
        return total();
    }
};

extern "C" void named_pairs(int n,const int *parent,const unsigned char *valid_pair,
    const unsigned char *visible,const double *stop,const double *ps,const double *pr,
    const double *x,const double *rx,const double *rdepth,const double *distance,
    int direction,int depth,int nearest,int same_word,double *out,double *reference) {
    Forest tree(n,parent);
    std::vector<double> subtract(n),orig(n);
    std::vector<int> order; order.reserve(n);
    for(int r=0;r<n;++r) {
        if(!visible[r] || !stop[r] || !pr[r]) continue;
        order.clear();
        for(int v=0;v<n;++v) {
            bool depth_ban=visible[v] && depth && depth*(rdepth[v]-rdepth[r])<0.;
            tree.emit[v]=stop[v]*(1.-(depth_ban?pr[v]:0.));
            tree.split[v]=1.-stop[v];
            subtract[v]=depth_ban && same_word?0.:stop[v]*ps[v];
            if(visible[v] && valid_pair[v*n+r] && distance[v*n+r]>0. && subtract[v]>0.)
                order.push_back(v);
        }
        // The selected reference must emit. Ancestors must split. Descendants have no effect.
        tree.emit[r]=stop[r]*pr[r];tree.split[r]=0.;
        for(int v=parent[r];v>=0;v=parent[v])tree.emit[v]=0.;
        for(int v=0;v<n;++v)tree.recalc(v);
        reference[r]=tree.total();
        if(reference[r]==0)continue;
        if(nearest)std::stable_sort(order.begin(),order.end(),[&](int a,int b){
            return distance[a*n+r]<distance[b*n+r];});
        size_t from=0;
        while(from<order.size()) {
            size_t end=from+1;
            if(nearest)while(end<order.size() && distance[order[from]*n+r]==distance[order[end]*n+r])++end;
            double before=tree.total();
            for(size_t k=from;k<end;++k) {
                int v=order[k];orig[v]=tree.emit[v];
                double after=tree.change(v,std::max(0.,orig[v]-subtract[v]));
                out[v*n+r]=std::max(0.,before-after);
                if(!nearest || end-from>1)tree.change(v,orig[v]);
            }
            if(nearest && end-from>1)
                for(size_t k=from;k<end;++k){int v=order[k];tree.change(v,std::max(0.,orig[v]-subtract[v]));}
            from=end;
        }
    }
}
