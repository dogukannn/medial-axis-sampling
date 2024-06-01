#include <igl/opengl/glfw/Viewer.h>
#include <random>

#include "igl/ray_mesh_intersect.h"
#include "igl/signed_distance.h"
#include "igl/readOBJ.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/measure.h>


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef K::Ray_3 Ray;
typedef K::Vector_3 Vector;
typedef CGAL::Surface_mesh<Point> Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor face_descriptor;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> Tree;
typedef Mesh::Face_index face_index;


igl::opengl::glfw::Viewer viewer;

double face_area(const Mesh& mesh, face_index fd) {
    return CGAL::Polygon_mesh_processing::face_area(fd, mesh);
}

Point random_point_in_triangle(const Point& a, const Point& b, const Point& c) {
    double r1 = CGAL::get_default_random().get_double();
    double r2 = CGAL::get_default_random().get_double();
    
    if (r1 + r2 > 1.0) {
        r1 = 1.0 - r1;
        r2 = 1.0 - r2;
    }

    double r3 = 1.0 - r1 - r2;
    return Point(r1 * a.x() + r2 * b.x() + r3 * c.x(),
                 r1 * a.y() + r2 * b.y() + r3 * c.y(),
                 r1 * a.z() + r2 * b.z() + r3 * c.z());
}

class MeshSampler {
public:
    MeshSampler(const Mesh& mesh) : mesh(mesh) {
        precompute_cumulative_areas();
    }

    std::pair<Point, face_index> sample_random_point() const {
        double r = CGAL::get_default_random().get_double(0.0, total_area);

        auto it = std::lower_bound(cumulative_areas.begin(), cumulative_areas.end(), r);
        face_index fd = face_indices[std::distance(cumulative_areas.begin(), it)];
        
        Mesh::Halfedge_index h = mesh.halfedge(fd);
        Point a = mesh.point(mesh.source(h));
        Point b = mesh.point(mesh.target(h));
        Point c = mesh.point(mesh.target(mesh.next(h)));
        
        Point p = random_point_in_triangle(a, b, c);
        
        return std::make_pair(p, fd);
    }

private:
    const Mesh& mesh;
    std::vector<double> cumulative_areas;
    std::vector<face_index> face_indices;
    double total_area = 0.0;

    void precompute_cumulative_areas() {
        cumulative_areas.reserve(mesh.number_of_faces());
        for (face_index fd : mesh.faces()) {
            total_area += face_area(mesh, fd);
            cumulative_areas.push_back(total_area);
            face_indices.push_back(fd); 
        }
    }
};


bool is_sphere_inside_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& center,
                           double radius)
{
    Eigen::VectorXd S;
    Eigen::VectorXi I;
    Eigen::MatrixXd C;
    Eigen::MatrixXd N;

    // Compute the signed distance from the sphere center to the mesh
    igl::signed_distance(center, V, F, igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL, S, I, C, N);

    // Check if the distance is negative and its absolute value is greater than or equal to the radius
    if (S(0) < 0 && std::abs(S(0)) >= radius - 0.01)
    {
        return true;
    }
    return false;
}

//sample random point from mesh
void SamplePoint(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::RowVector3d& point, int& face_index)
{
    int num_vertices = V.rows();
    int num_faces = F.rows();


    {
        //sample random face
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, num_faces - 1);
        face_index = distrib(gen);
    }


    auto face = F.row(face_index);

    //get face points
    Eigen::RowVector3d p1 = V.row(face(0));
    Eigen::RowVector3d p2 = V.row(face(1));
    Eigen::RowVector3d p3 = V.row(face(2));

    std::random_device rd; // Seed for random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0.0, 1.0); // Uniform distribution between 0 and 1

    double u = dis(gen);
    double v = dis(gen);

    if (u + v > 1.0)
    {
        u = 1.0 - u;
        v = 1.0 - v;
    }
    point = (1 - u - v) * p1 + u * p2 + v * p3;
}

template <typename T>
auto dist(T const& V, T const& W)
{
    auto const slen = (V-W).squared_length();
    auto const d = CGAL::approximate_sqrt(slen);
    return d;
}


bool IsSphereInsideMesh(const Mesh& mesh, Tree& tree, Vector center, double radius)
{
    Point centerPoint = Point(center.x(), center.y(), center.z());
    auto closestPoint = tree.closest_point(centerPoint);
    double sd = CGAL::squared_distance(centerPoint, closestPoint);
    double d = CGAL::approximate_sqrt(sd);
    if(d >= (radius))
    {
        return true;
    }
    return false;
}

Vector CalculateMedialPoint(const Mesh& mesh, Tree& tree, Vector p, Vector q)
{
    auto midPoint = (p + q) / 2;
    auto basePoint = p;
    double radius = dist(midPoint,p);
    
    while(dist(q,p) > 0.0001)
    {
        if (IsSphereInsideMesh(mesh, tree, midPoint, radius))
            p = midPoint;
        else
            q = midPoint;
    
        midPoint = (p + q) / 2;
        radius = dist(midPoint, basePoint);
    }
    auto medialPoint = midPoint;

    //convert to eigen point and add to viewer
    Eigen::RowVector3d medialPointE = Eigen::RowVector3d(medialPoint.x(), medialPoint.y(), medialPoint.z());
    viewer.data().add_points(medialPointE, Eigen::RowVector3d((medialPoint.x() + 1.) / 2., (medialPoint.y()+1.)/2., (medialPoint.z()+1.)/2.));
    
    return medialPoint;
}

void CalculateMedialPoint(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, Eigen::RowVector3d P,
                          Eigen::RowVector3d Q, Eigen::RowVector3d& medialPoint)
{
    Eigen::RowVector3d midPoint = (P + Q) / 2;

    auto basePoint = P;
    double radius = (midPoint - P).norm();

    while ((Q - P).norm() > 0.0001)
    {
        //Eigen::Vector3d mpt = midPoint.transpose();
        if (is_sphere_inside_mesh(V, F, midPoint, radius))
            P = midPoint;
        else
            Q = midPoint;

        midPoint = (P + Q) / 2;
        radius = (midPoint - basePoint).norm();
    }

    medialPoint = midPoint;

    viewer.data().add_points(medialPoint, Eigen::RowVector3d(1, 0, 0));
    //viewer.data().add_edges(basePoint, medialPoint, Eigen::RowVector3d(0, 1, 0));
}


void GetMedialAxisPoints(const Mesh& mesh, MeshSampler& sampler, Tree& tree)
{
    //Eigen::MatrixXd medial_axis_points;
    // Implement your code here
    int num_samples = 5000;
    for (int i = 0; i < num_samples; i++)
    {
        //Eigen::RowVector3d point;
        //int face_index = 0;
        //SamplePoint(V, F, point, face_index);
        //auto [point, face_index] = sampler.sample_random_point();
        auto sample_pair = sampler.sample_random_point();
        auto point = sample_pair.first;
        auto face_index = sample_pair.second;

        Eigen::RowVector3d pe = Eigen::RowVector3d(point.x(), point.y(), point.z());
        //viewer.data().add_points(pe, Eigen::RowVector3d(1, 0, 0));

        // Compute the inward normal
        Vector face_normal = CGAL::Polygon_mesh_processing::compute_face_normal(face_index, mesh);
        Vector inward_normal = -face_normal;

        // Create the ray
        Ray ray(point + inward_normal * 0.01, inward_normal);

        // Intersect the ray with the mesh
        auto intersection = tree.first_intersection(ray);
        if (intersection)
        {
            
            if (const Point* p =  boost::get<Point>(&(intersection->first)))
            {
                //add it to the wiever with yellow color
                Eigen::RowVector3d intersection_point = Eigen::RowVector3d(p->x(), p->y(), p->z());
                //viewer.data().add_points(intersection_point, Eigen::RowVector3d(1, 1, 0));
                //add edge between the point and intersection point
                //viewer.data().add_edges(pe, intersection_point, Eigen::RowVector3d(0, 1, 0));

                Vector pointVector = point - CGAL::ORIGIN;
                Vector intersectionVector = *p - CGAL::ORIGIN;
                CalculateMedialPoint(mesh, tree, pointVector, intersectionVector);
            }
            else
            {
                std::cout << "Intersection at some primitive." << std::endl;
            }
        }
        else
        {
            std::cout << "No intersection found." << std::endl;
        }

        
        //igl::Hit hit;
        //igl::ray_mesh_intersect(point + (-FN.row(face_index) * 0.01), -FN.row(face_index), V, F, hit);
        //
        //if (hit.t > 0)
        //{
        //    Eigen::RowVector3d hit_point = point + (-FN.row(face_index) * 0.01) + hit.t * -FN.row(face_index);
        //
        //    Eigen::RowVector3d medialPoint;
        //    CalculateMedialPoint(V, F, point, hit_point, medialPoint);
        //}
        //else
        //{
        //    std::cout << "No intersection found" << std::endl;
        //}
    }

    //return medial_axis_points;
}

int main(int argc, char* argv[])
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd FN;
    igl::readOBJ("../Assets/cube.obj", V, F);
    igl::per_face_normals(V, F, FN);

    //GetMedialAxisPoints(V, F, FN);

    // Convert igl mesh to CGAL mesh
    Mesh cgal_mesh;
    for (int i = 0; i < V.rows(); ++i)
    {
        cgal_mesh.add_vertex(Point(V(i, 0), V(i, 1), V(i, 2)));
    }

    for (int i = 0; i < F.rows(); ++i)
    {
        Mesh::Vertex_index v1 = Mesh::Vertex_index(F(i, 0));
        Mesh::Vertex_index v2 = Mesh::Vertex_index(F(i, 1));
        Mesh::Vertex_index v3 = Mesh::Vertex_index(F(i, 2));
        cgal_mesh.add_face(v1, v2, v3);
    }

    // Create the AABB tree
    Tree tree(faces(cgal_mesh).first, faces(cgal_mesh).second, cgal_mesh);
    tree.accelerate_distance_queries();

    MeshSampler sampler(cgal_mesh);

    GetMedialAxisPoints(cgal_mesh, sampler, tree);

    viewer.data().point_size = 3.0f;
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    viewer.launch();
}
