import unittest

from ref.wen_map import convert, robots_allows

KML = b"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>M</name>
<description>&#27489;&#36814;&#22823;&#23478;&#21462;&#29992;</description>
<Folder><name>NAVY</name>
<Placemark><name>A</name><description><![CDATA[x<br>y<img src="z">]]></description>
<Point><coordinates>110.44,21.23,0</coordinates></Point></Placemark>
<Placemark><name>P</name><Polygon><outerBoundaryIs><LinearRing>
<coordinates>1,2,0 3,4,0 1,2,0</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>
</Folder></Document></kml>"""

ROBOTS = """User-agent: *
Disallow: /maps/
Allow: /maps/d/
Disallow: /x/*.kml$
"""


class T(unittest.TestCase):
    def test_convert(self):
        fc = convert(KML, "2026-10-31T00:00:00Z")
        self.assertEqual(fc["meta"]["feature_count"], 2)
        self.assertEqual(fc["meta"]["map_description"], "歡迎大家取用")
        a = fc["features"][0]
        self.assertEqual(a["geometry"], {"type": "Point", "coordinates": [110.44, 21.23]})
        self.assertEqual(a["properties"], {"name": "A", "description": "x\ny", "folder": "NAVY"})
        self.assertEqual(fc["features"][1]["geometry"]["type"], "Polygon")

    def test_robots_longest_match(self):
        self.assertTrue(robots_allows(ROBOTS, "/maps/d/kml?mid=1")[0])
        self.assertFalse(robots_allows(ROBOTS, "/maps/place")[0])
        self.assertFalse(robots_allows(ROBOTS, "/x/a.kml")[0])
        self.assertTrue(robots_allows(ROBOTS, "/x/a.kml?q")[0])


if __name__ == "__main__":
    unittest.main()
