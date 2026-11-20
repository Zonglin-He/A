#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {
constexpr double half_pi = 1.57079632679489661923;
constexpr double pi = 2.0 * half_pi;

double length_overlap(double a, double b, double separation) {
    return std::max(0., std::min(std::min(a, b), .5*(a+b)-std::abs(separation)));
}

// Mean of the intersection area of two parallel rectangles with a shared uniform yaw.
double shared_area(double length_a, double width_a, double length_b, double width_b, double r) {
    const double lx = .5*(length_a+length_b), ly = .5*(width_a+width_b);
    if (r*r >= lx*lx+ly*ly) return 0.;
    const double sx = std::min(length_a,length_b), sy = std::min(width_a,width_b);
    if (r <= std::min(lx-sx,ly-sy)) return sx*sy;
    std::array<double,6> knots{0.,half_pi};
    int n = 2;
    for (double x : {lx, lx-sx}) if (x>0 && x<r) knots[n++] = std::acos(x/r);
    for (double y : {ly, ly-sy}) if (y>0 && y<r) knots[n++] = std::asin(y/r);
    std::sort(knots.begin(),knots.begin()+n);
    double total = 0.;
    for (int i=1; i<n; ++i) {
        const double a=knots[i-1], b=knots[i], t=.5*(a+b);
        const double x=r*std::cos(t), y=r*std::sin(t);
        if (b<=a || x>=lx || y>=ly) continue;
        const bool outer_x=x>lx-sx, outer_y=y>ly-sy;
        const double A=outer_x?lx:sx, B=outer_x?-r:0.;
        const double C=outer_y?ly:sy, D=outer_y?-r:0.;
        const double sin_a=std::sin(a), sin_b=std::sin(b);
        total += A*C*(b-a) + B*C*(sin_b-sin_a)
               + A*D*(std::cos(a)-std::cos(b)) + .5*B*D*(sin_b*sin_b-sin_a*sin_a);
    }
    return std::clamp(total/half_pi,0.,sx*sy);
}

double disk_area(double a, double b, double d) {
    if (d>=a+b) return 0.;
    if (d<=std::abs(a-b)) return pi*std::min(a*a,b*b);
    const double x=(a*a-b*b+d*d)/(2*d), y=d-x;
    const double h=std::sqrt(std::max(0.,a*a-x*x));
    return a*a*std::acos(std::clamp(x/a,-1.,1.)) + b*b*std::acos(std::clamp(y/b,-1.,1.)) - d*h;
}

struct Point { double x,y; };
struct Rect {
    double x,y,z,l,w,h,c,s,volume,radius,disk;
    explicit Rect(const double* p):
        x(p[0]),y(p[1]),z(p[2]),l(std::max(0.,p[3])),w(std::max(0.,p[4])),h(std::max(0.,p[5])),
        c(std::cos(p[6])),s(std::sin(p[6])),volume(l*w*h),radius(.5*std::hypot(l,w)),disk(std::sqrt(l*w/pi)) {}
};

// Clip rectangle b against four half-planes in rectangle a's local coordinate system.
double measured_area(const Rect& a,const Rect& b,double dx,double dy) {
    const double ux=a.c*dx+a.s*dy, uy=-a.s*dx+a.c*dy;
    const double cs=a.c*b.c+a.s*b.s, sn=a.c*b.s-a.s*b.c;
    std::array<Point,12> points{}, next{};
    int n=0;
    for (Point sign: {Point{-1.,-1.},Point{1.,-1.},Point{1.,1.},Point{-1.,1.}}) {
        const double x=.5*b.l*sign.x,y=.5*b.w*sign.y;
        points[n++]={ux+cs*x-sn*y,uy+sn*x+cs*y};
    }
    for (int side=0; side<4 && n>=3; ++side) {
        bool horizontal=side<2;
        double limit=.5*(horizontal?a.l:a.w), sign=side%2==0?1.:-1.;
        int count=0;
        for (int i=0; i<n; ++i) {
            const Point p=points[(i+n-1)%n],q=points[i];
            const double dp=limit-sign*(horizontal?p.x:p.y),dq=limit-sign*(horizontal?q.x:q.y);
            if ((dp>=0.)!=(dq>=0.)) {
                const double t=dp/(dp-dq);
                next[count++]={p.x+t*(q.x-p.x),p.y+t*(q.y-p.y)};
            }
            if (dq>=0.) next[count++]=q;
        }
        points=next; n=count;
    }
    double area=0.;
    for(int i=0;i<n;++i) {
        const Point p=points[i],q=points[(i+1)%n];
        area+=p.x*q.y-p.y*q.x;
    }
    return .5*std::abs(area);
}
}

extern "C" void kernels(const double* boxes, int n, double* values) {
    const int stride=n*n;
    std::fill(values,values+5*stride,0.);
    std::vector<Rect> rects;
    rects.reserve(n);
    for (int i=0;i<n;++i) rects.emplace_back(boxes+7*i);
    for(int i=0;i<n;++i) {
        const Rect& a=rects[i];
        for(int k=0;k<5;++k) values[k*stride+i*n+i]=1.;
        for(int j=0;j<i;++j) {
            const Rect& b=rects[j];
            const double dz=length_overlap(a.h,b.h,b.z-a.z);
            if (dz<=0. || a.volume<=0. || b.volume<=0.) continue;
            const double dx=b.x-a.x,dy=b.y-a.y,r=std::hypot(dx,dy);
            if (r>=a.radius+b.radius) continue;
            const double invsqrt2=std::sqrt(.5);
            const std::array<double,5> areas={
                shared_area(a.l,a.w,b.l,b.w,r),measured_area(a,b,dx,dy),disk_area(a.disk,b.disk,r),
                shared_area(a.l,a.w,b.w,b.l,r),
                length_overlap(a.l,b.l,(dx+dy)*invsqrt2)*length_overlap(a.w,b.w,(dx-dy)*invsqrt2)
            };
            for(int k=0;k<5;++k) {
                const double intersection=std::clamp(areas[k]*dz,0.,std::min(a.volume,b.volume));
                const double u=intersection/(a.volume+b.volume-intersection);
                values[k*stride+i*n+j]=values[k*stride+j*n+i]=u;
            }
        }
    }
}
