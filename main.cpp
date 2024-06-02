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
#include <CGAL/boost/graph/graph_traits_HalfedgeDS_default.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/squared_distance_3.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef K::Ray_3 Ray;
typedef K::Vector_3 Vector;
typedef K::Segment_3 Segment;
typedef CGAL::Surface_mesh<Point> Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor face_descriptor;
typedef CGAL::AABB_face_graph_triangle_primitive<Mesh> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> Tree;
typedef Mesh::Face_index face_index;
typedef boost::optional< Tree::Intersection_and_primitive_id<Triangle>::Type > Triangle_intersection;

igl::opengl::glfw::Viewer viewer;

#define SAMPLE_PER_EDGE 5
#define TOTAL_SAMPLES 5000

struct MedialPoint
{
    Vector point;
    std::set<CGAL::SM_Face_index> governor_faces;
};

// structure to hold closest medial axis point, distance to it, and face id


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
    if(d >= (radius - 0.01))
    {
        return true;
    }
    return false;
}

std::vector<Vector> GetEdgeNormals(const Mesh& mesh, const std::vector<Vector>& face_normals)
{
    std::vector<Vector> edge_normals;
    for(auto e : edges(mesh))
    {
        auto h = halfedge(e, mesh);
        auto hf = opposite(h, mesh);

        //get normal of f1
        auto f1 = face(h, mesh);
        auto f1Area = CGAL::Polygon_mesh_processing::face_area(f1, mesh);
        auto f1Normal = face_normals[f1] * f1Area;
        
        //get normal of f2
        auto f2 = face(hf, mesh);
        if(f2.idx() == UINT32_MAX)
        {
            edge_normals.push_back(f1Normal);
            continue;
        }
        auto f2Area = CGAL::Polygon_mesh_processing::face_area(f2, mesh);
        auto f2Normal = face_normals[f2] * f2Area;

        //get edge normal
        auto edge_normal = (f1Normal + f2Normal) / (f1Area + f2Area);
        
        edge_normals.push_back(edge_normal);
    }
    return edge_normals;
}

Vector CalculateMedialPoint(const Mesh& mesh, Tree& tree, Vector p, Vector q)
{
    auto midPoint = (p + q) / 2;
    auto basePoint = p;
    double radius = dist(midPoint,p);
    
    while(dist(q,p) > 0.01)
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
    //viewer.data().add_points(medialPointE, Eigen::RowVector3d((medialPoint.x() + 1.) / 2., (medialPoint.y()+1.)/2., (medialPoint.z()+1.)/2.));
    
    return medialPoint;
}

void GetMedialAxisPoints(const Mesh& mesh, MeshSampler& sampler, Tree& tree)
{
    //Eigen::MatrixXd medial_axis_points;
    // Implement your code here
    int num_samples = 5000;
    for (int i = 0; i < num_samples; i++)
    {
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
                //std::cout << "Intersection at some primitive." << std::endl;
            }
        }
        else
        {
            //std::cout << "No intersection found." << std::endl;
        }
    }
}

Vector CalculateEffectiveNormal(const Mesh& mesh, const Tree& tree, const std::vector<Vector>& face_normals, Vector pv)
{
    Point centerPoint = Point(pv.x(), pv.y(), pv.z());
    auto closestPoint = tree.closest_point(centerPoint);
    double sd = CGAL::squared_distance(centerPoint, closestPoint);
    double radius = CGAL::approximate_sqrt(sd);
    radius = radius + 0.001;
    
    CGAL::Bbox_3 query(
    centerPoint.x() - radius, 
    centerPoint.y() - radius, 
    centerPoint.z() - radius, 
    centerPoint.x() + radius, 
    centerPoint.y() + radius, 
    centerPoint.z() + radius);
    
    std::list<Tree::Primitive_id> intersections;
    tree.all_intersected_primitives(query, std::back_inserter(intersections));
    
    std::vector<Triangle> intersected_facets;
    std::vector<Point> intersected_points;
    std::vector<Segment> intersected_edges;
    Vector effectiveNormal = Vector(0.0,0.0,0.0);
    Vector lastNormal = Vector(0.0,0.0,0.0);
    for (auto it = intersections.begin(); it != intersections.end(); it++)
    {
        CGAL::SM_Face_index fi = boost::get<CGAL::SM_Face_index>(*it);
        auto vertices = mesh.vertices_around_face(mesh.halfedge(fi));
        auto itv = vertices.begin();
        auto v1 = mesh.point(*itv);
        itv++;
        auto v2 = mesh.point(*itv);
        itv++;
        auto v3 = mesh.point(*itv);
        Triangle t(v1,v2,v3);
        
        auto sd = squared_distance(t, centerPoint);
        auto d = CGAL::approximate_sqrt(sd);
        if( d > radius - 0.002 && d < radius + 0.001)
        {
            intersected_facets.push_back(t);
        }
        effectiveNormal += face_normals[fi];
        lastNormal = face_normals[fi];
    }
    effectiveNormal = effectiveNormal / intersected_facets.size();
    
    Point efnp = Point(effectiveNormal.x(), effectiveNormal.y(), effectiveNormal.z());
    Point origin = Point(0.0, 0.0, 0.0);
    if(squared_distance(efnp, origin) < 0.0001)
    {
        effectiveNormal = lastNormal;
    }
    //draw effective normal
    //viewer.data().add_points(Eigen::RowVector3d(pv.x(), pv.y(), pv.z()), Eigen::RowVector3d(1, 0, 0));
    //viewer.data().add_edges(Eigen::RowVector3d(pv.x(), pv.y(), pv.z()), Eigen::RowVector3d(pv.x() + effectiveNormal.x(), pv.y() + effectiveNormal.y(), pv.z() + effectiveNormal.z()), Eigen::RowVector3d(1, 0, 1));

    effectiveNormal = effectiveNormal / sqrt(effectiveNormal.squared_length());
    return effectiveNormal;
}


Vector ComputeJuntionPoint(const Mesh& mesh, Tree& tree, Vector p, Vector q, std::vector<Vector>& faceNormals)
{
    auto midPoint = (p + q) / 2;
    Vector junction;
    
    while(dist(q,p) > 0.01)
    {
        Point basePoint = tree.closest_point(Point(midPoint.x(), midPoint.y(), midPoint.z()));
        auto pf_pair = tree.closest_point_and_primitive(Point(midPoint.x(), midPoint.y(), midPoint.z()));
        auto fi = boost::get<CGAL::SM_Face_index>(pf_pair.second);
        
        Vector faceNormal = faceNormals[fi.idx()];
        
        Vector inwardFaceNormal = -faceNormal;
        Ray ray(basePoint + inwardFaceNormal * 0.01, inwardFaceNormal);

        auto intersection = tree.first_intersection(ray);
        if (intersection)
        {
            if (const Point* pp =  boost::get<Point>(&(intersection->first)))
            {
                Vector pointVector = basePoint - CGAL::ORIGIN;
                Vector intersectionVector = *pp - CGAL::ORIGIN;
                auto junctionPoint = CalculateMedialPoint(mesh, tree, pointVector, intersectionVector);
                auto junctionEffectiveNormal = CalculateEffectiveNormal(mesh, tree, faceNormals, junctionPoint);

                auto pEffectiveNormal = CalculateEffectiveNormal(mesh, tree, faceNormals, p);
                auto qEffectiveNormal = CalculateEffectiveNormal(mesh, tree, faceNormals, q);

                auto np = cross_product(junctionEffectiveNormal, pEffectiveNormal).squared_length();
                if(np < 0.001)
                    p = midPoint;
                else
                {
                    auto nq = cross_product(junctionEffectiveNormal, qEffectiveNormal).squared_length();
                    if(nq < 0.001)
                        q = midPoint;
                    else
                    {
                        Eigen::RowVector3d junctionPointE = Eigen::RowVector3d(junctionPoint.x(), junctionPoint.y(), junctionPoint.z());
                        viewer.data().add_points(junctionPointE, Eigen::RowVector3d(1,1,1));
                        return midPoint;
                    }
                }
                
            }
        }
        midPoint = (p + q) / 2;
    }
    auto junctionPoint = midPoint;

    //convert to eigen point and add to viewer
    Eigen::RowVector3d junctionPointE = Eigen::RowVector3d(junctionPoint.x(), junctionPoint.y(), junctionPoint.z());
    viewer.data().add_points(junctionPointE, Eigen::RowVector3d(1,1,1));
    
    return junctionPoint;
}

void ComputeMedialJunctionPoints(const Mesh& mesh, MeshSampler& sampler, Tree& tree, std::vector<Vector>& face_normals)
{
    std::vector<Vector> medialAxisPoints;
    std::vector<Vector> medialJunctionPoints;
    
    for (int i = 0; i < TOTAL_SAMPLES; i++)
    {
        auto sample_pair = sampler.sample_random_point();
        auto point = sample_pair.first;
        auto face_index = sample_pair.second;

        Eigen::RowVector3d pe = Eigen::RowVector3d(point.x(), point.y(), point.z());

        Vector face_normal = CGAL::Polygon_mesh_processing::compute_face_normal(face_index, mesh);
        Vector inward_normal = -face_normal;

        // Create the ray
        Ray ray(point + inward_normal * 0.01, inward_normal);

        auto intersection = tree.first_intersection(ray);
        if (intersection)
        {
            if (const Point* p = boost::get<Point>(&(intersection->first)))
            {
                //add it to the wiever with yellow color
                Eigen::RowVector3d intersection_point = Eigen::RowVector3d(p->x(), p->y(), p->z());
                //viewer.data().add_points(intersection_point, Eigen::RowVector3d(1, 1, 0));
                //add edge between the point and intersection point
                //viewer.data().add_edges(pe, intersection_point, Eigen::RowVector3d(0, 1, 0));

                Vector pointVector = point - CGAL::ORIGIN;
                Vector intersectionVector = *p - CGAL::ORIGIN;
                auto medialPoint = CalculateMedialPoint(mesh, tree, pointVector, intersectionVector);
                medialAxisPoints.push_back(medialPoint);

                //add medial point to the viewer
                Eigen::RowVector3d medialPointE = Eigen::RowVector3d(medialPoint.x(), medialPoint.y(), medialPoint.z());
                viewer.data().add_points(medialPointE, Eigen::RowVector3d((medialPoint.x() + 1.) / 2., (medialPoint.y()+1.)/2., (medialPoint.z()+1.)/2.));
                
                //if (medialAxisPoints.size() > 1)
                //{
                //    Vector v1 = medialAxisPoints[medialAxisPoints.size() - 2];
                //    Vector v2 = medialAxisPoints[medialAxisPoints.size() - 1];
                //
                //    Vector effectiveNormalv1 = CalculateEffectiveNormal(mesh, tree, face_normals, v1);
                //    Vector effectiveNormalv2 = CalculateEffectiveNormal(mesh, tree, face_normals, v2);
                //
                //    double n = cross_product(effectiveNormalv1, effectiveNormalv2).squared_length();
                //    if (n > 0.1)
                //    {
                //        Vector junctionPoint = ComputeJuntionPoint(mesh, tree, v1, v2, face_normals);
                //        medialJunctionPoints.push_back(junctionPoint);
                //    }
                //    else
                //    {
                //        std::cout << "skipped " << n << std::endl;
                //    }
                //}
            }
            else
            {
                //std::cout << "Intersection at some primitive." << std::endl;
            }
        }
        else
        {
            //std::cout << "No intersection found." << std::endl;
        }
    }

}

int main(int argc, char* argv[])
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd FN;
    igl::readOBJ("../Assets/dragon.obj", V, F);

    // Convert igl mesh to CGAL mesh
    Mesh cgal_mesh;
    for (int i = 0; i < V.rows(); ++i)
    {
        Point x = Point(V(i, 0) * 5.0, V(i, 1) * 5.0, V(i, 2) * 5.0);
        cgal_mesh.add_vertex(x);
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

    std::vector<Vector> face_normals;
    face_normals.reserve(cgal_mesh.number_of_faces());
    for (auto f : faces(cgal_mesh))
    {
        face_normals.push_back(CGAL::Polygon_mesh_processing::compute_face_normal(f, cgal_mesh));
    }

    ComputeMedialJunctionPoints(cgal_mesh, sampler, tree, face_normals);
    
    //GetMedialAxisPoints(cgal_mesh, sampler, tree);

    viewer.data().point_size = 3.0f;
    viewer.data().set_mesh(V, F);
    viewer.data().set_face_based(true);
    viewer.launch();
}
