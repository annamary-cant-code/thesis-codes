import xml.etree.ElementTree as ET
from collections import Counter

OSM_FILE = "map_full.osm"

# Parse the XML
tree = ET.parse(OSM_FILE)
root = tree.getroot()

# Collect all tags from ways and relations (these define areas/polygons)
way_tags = Counter()
# relation_tags = Counter()

for way in root.findall("way"):
    for tag in way.findall("tag"):
        k = tag.get("k")
        v = tag.get("v")
        way_tags[f"{k}={v}"] += 1

# Print ALL values for the specific keys we care about
TARGET_KEYS = {"building", "highway", "landuse", "natural", "leisure", "railway", "waterway"}


# for rel in root.findall("relation"):
#     for tag in rel.findall("tag"):
#         k = tag.get("k")
#         v = tag.get("v")
#         relation_tags[f"{k}={v}"] += 1
# 
# # Print the most common ones
# print("=== TOP 60 WAY TAGS ===")
# for tag, count in way_tags.most_common(60):
#     print(f"  {count:5d}  {tag}")
# 
# print("\n=== TOP 30 RELATION TAGS ===")
# for tag, count in relation_tags.most_common(30):
#     print(f"  {count:5d}  {tag}")


print("=== TAGS BY KEY (sorted by key then count) ===")
by_key = {}
for tag, count in way_tags.items():
    k = tag.split("=")[0]
    if k in TARGET_KEYS:
        if k not in by_key:
            by_key[k] = []
        by_key[k].append((count, tag))

for key in sorted(by_key.keys()):
    print(f"\n--- {key} ---")
    for count, tag in sorted(by_key[key], reverse=True):
        print(f"  {count:5d}  {tag}")

