from shapely.geometry import GeometryCollection, Polygon

from sahi.utils.shapely import MultiPolygon, ShapelyAnnotation, get_shapely_box, get_shapely_multipolygon


class TestShapelyUtils:
    def test_get_shapely_box(self):
        x, y, width, height = 1, 1, 256, 256
        shapely_box = get_shapely_box(x, y, width, height)

        assert shapely_box.exterior.coords.xy[0].tolist() == [257.0, 257.0, 1.0, 1.0, 257.0]
        assert shapely_box.area == 65536
        assert shapely_box.bounds == (1, 1, 257, 257)

    def test_get_shapely_multipolygon(self):
        coco_segmentation = [[1, 1, 325, 125, 250, 200, 5, 200]]
        shapely_multipolygon = get_shapely_multipolygon(coco_segmentation)

        assert shapely_multipolygon.geoms[0].exterior.coords.xy[0].tolist() == [1.0, 325, 250, 5, 1]
        assert shapely_multipolygon.area == 41177.5
        assert shapely_multipolygon.bounds == (1, 1, 325, 200)

    def test_get_shapely_multipolygon_naughty(self):
        # self-intersection case
        coco_segmentation = [[3559.0, 2046.86, 3.49, 2060.0, 3540.9, 3249.7, 2060.0, 3239.61, 2052.87]]
        shapely_multipolygon = get_shapely_multipolygon(coco_segmentation)
        assert shapely_multipolygon.is_valid

    def test_shapely_annotation(self):
        # init shapely_annotation from coco segmentation
        segmentation = [[1, 1, 325, 125.2, 250, 200, 5, 200]]
        shapely_multipolygon = get_shapely_multipolygon(segmentation)
        shapely_annotation = ShapelyAnnotation.from_coco_segmentation(segmentation)

        # test conversion methods
        coco_segmentation = shapely_annotation.to_coco_segmentation()
        assert coco_segmentation == [[1, 1, 325, 125, 250, 200, 5, 200]]
        opencv_contours = shapely_annotation.to_opencv_contours()
        assert opencv_contours == [
            [
                [[1, 1]],
                [[325, 125]],
                [[250, 200]],
                [[5, 200]],
                [[1, 1]],
            ]
        ]
        coco_bbox = shapely_annotation.to_xywh()
        assert coco_bbox == [1, 1, 324, 199]
        voc_bbox = shapely_annotation.to_xyxy()
        assert voc_bbox == [1, 1, 325, 200]

        # test properties
        assert shapely_annotation.area == int(shapely_multipolygon.area)
        assert shapely_annotation.multipolygon == shapely_multipolygon

        # init shapely_annotation from coco bbox
        coco_bbox = [1, 1, 100, 100]
        shapely_polygon = get_shapely_box(x=coco_bbox[0], y=coco_bbox[1], width=coco_bbox[2], height=coco_bbox[3])
        shapely_annotation = ShapelyAnnotation.from_coco_bbox(coco_bbox)

        # test conversion methods
        coco_segmentation = shapely_annotation.to_coco_segmentation()
        assert coco_segmentation == [[101, 1, 101, 101, 1, 101, 1, 1]]
        opencv_contours = shapely_annotation.to_opencv_contours()
        assert opencv_contours == [
            [
                [[101, 1]],
                [[101, 101]],
                [[1, 101]],
                [[1, 1]],
                [[101, 1]],
            ]
        ]
        coco_bbox = shapely_annotation.to_xywh()
        assert coco_bbox == [1, 1, 100, 100]
        voc_bbox = shapely_annotation.to_xyxy()
        assert voc_bbox == [1, 1, 101, 101]

        # test properties
        assert shapely_annotation.area == MultiPolygon([shapely_polygon]).area
        assert shapely_annotation.multipolygon == MultiPolygon([shapely_polygon])

    def test_get_intersection(self):
        x, y, width, height = 1, 1, 256, 256
        shapely_box = get_shapely_box(x, y, width, height)

        coco_segmentation = [[1, 1, 325, 125, 250, 200, 5, 200]]
        shapely_annotation = ShapelyAnnotation.from_coco_segmentation(coco_segmentation)

        intersection_shapely_annotation = shapely_annotation.get_intersection(shapely_box)

        test_list = intersection_shapely_annotation.to_list()[0]
        true_list = [(0, 0), (4, 199), (249, 199), (256, 192), (256, 97), (0, 0)]
        for i in range(len(test_list)):
            for j in range(2):
                assert int(test_list[i][j]) == int(true_list[i][j])

        assert intersection_shapely_annotation.to_coco_segmentation() == [
            [
                0,
                0,
                4,
                199,
                249,
                199,
                256,
                192,
                256,
                97,
            ]
        ]

        assert intersection_shapely_annotation.to_xywh() == [0, 0, 256, 199]

        assert intersection_shapely_annotation.to_xyxy() == [0, 0, 256, 199]

    def test_get_empty_intersection(self):
        x, y, width, height = 300, 300, 256, 256
        shapely_box = get_shapely_box(x, y, width, height)

        coco_segmentation = [[1, 1, 325, 125, 250, 200, 5, 200]]
        shapely_annotation = ShapelyAnnotation.from_coco_segmentation(coco_segmentation)

        intersection_shapely_annotation = shapely_annotation.get_intersection(shapely_box)

        assert intersection_shapely_annotation.area == 0

        assert intersection_shapely_annotation.to_xywh() == []

    def test_get_shapely_multipolygon_make_valid_returns_geometry_collection(self, monkeypatch):
        # Prepare simple polygons to be returned by make_valid inside a GeometryCollection
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        multi = MultiPolygon([poly2])

        # Monkeypatch make_valid to return a GeometryCollection containing a Polygon and a MultiPolygon
        import sahi.utils.shapely as shapely_module

        def fake_make_valid(_):
            return GeometryCollection([poly1, multi])

        monkeypatch.setattr(shapely_module, "make_valid", fake_make_valid)

        # Use a segmentation that produces an invalid polygon so make_valid is invoked
        naughty_segmentation = [[3559.0, 2046.86, 3.49, 2060.0, 3540.9, 3249.7, 2060.0, 3239.61, 2052.87]]
        result = get_shapely_multipolygon(naughty_segmentation)

        # Result should be a MultiPolygon and area equals sum of component areas
        assert isinstance(result, MultiPolygon)
        expected_area = poly1.area + poly2.area
        # allow small floating point tolerance
        assert abs(result.area - expected_area) < 1e-6
