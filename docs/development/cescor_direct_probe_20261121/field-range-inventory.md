# Illumination field inventory

Column z is in mm from the model origin. The 300 kV gun exit is at 450 mm. The incident calculation ends at 1599.1999975 mm. Values at that endpoint describe only the incident side. The original excitation and the 100 µm C2 aperture define the working point. [Mechanical interpretation and primary references](field-range-audit.md)

Each interval is a connected stretch above the stated fraction of its own channel maximum. The thresholds are reporting choices. Weak fields continue outside the reported intervals. A round channel uses on-axis longitudinal field, a dipole uses transverse field, and a multipole uses its leading transverse coefficient at finite radius. The last column lists other excited channels above 10% of their respective maxima at this component's center. Percentages from different rows cannot be added.

| Channel | Center | Mechanical interval | Field used | ≥50% | ≥10% | Other excited channels at center |
| --- | ---: | --- | --- | --- | --- | --- |
| C1 | 500.00 | 450.0–550.0 | Vacuum winding field | 450.0–538.5 | 450.0–609.2 | C2 23% |
| C2 | 650.00 | 550.0–750.0 | Vacuum winding field | 532.7–611.4 | 461.7–633.1, 660.1–704.4 | — |
| Condenser deflector upper X | 800.00 | 790.0–830.0 | Off, per A in vacuum | 792.0–808.0 | 784.9–815.1 | — |
| Condenser deflector upper Y | 800.00 | 790.0–830.0 | Off, per A in vacuum | 789.0–811.0 | 779.0–821.0 | — |
| Condenser deflector lower X | 820.00 | 790.0–830.0 | Off, per A in vacuum | 812.0–828.0 | 804.9–835.1 | — |
| Condenser deflector lower Y | 820.00 | 790.0–830.0 | Off, per A in vacuum | 809.0–831.0 | 799.0–841.0 | — |
| C3 | 915.00 | 830.0–1000.0 | Vacuum winding field | 950.2–1028.4 | 850.7–864.7, 930.5–1099.1 | — |
| Beam deflector upper X | 1027.50 | 1020.0–1060.0 | Off, per A in vacuum | 1018.4–1036.6 | 1011.1–1043.9 | C3 51%, ADL 10% |
| Beam deflector upper Y | 1027.50 | 1020.0–1060.0 | Off, per A in vacuum | 1015.8–1039.2 | 1005.6–1049.4 | C3 51%, ADL 10% |
| Beam deflector lower X | 1052.50 | 1020.0–1060.0 | Off, per A in vacuum | 1043.4–1061.6 | 1036.1–1068.9 | C3 28%, ADL 24% |
| Beam deflector lower Y | 1052.50 | 1020.0–1060.0 | Off, per A in vacuum | 1040.8–1064.2 | 1030.6–1074.4 | C3 28%, ADL 24% |
| ADL | 1088.20 | 1070.0–1106.4 | Vacuum winding field | 1068.9–1128.1 | 1026.9–1176.1 | C3 13% |
| QPH2 X | 1128.15 | 1120.9–1135.4 | Off, per A in vacuum | 1118.1–1138.2 | 1109.1–1147.2 | HP2 51%, ADL 50% |
| QPH2 Y | 1128.15 | 1120.9–1135.4 | Off, per A in vacuum | 1118.1–1138.2 | 1109.1–1147.2 | HP2 51%, ADL 50% |
| HP2 | 1144.50 | 1129.5–1159.5 | Material response to current | 1128.1–1160.4 | 1125.4–1163.2 | ADL 28% |
| DPH2 X | 1144.50 | 1137.2–1151.8 | Off, per A in vacuum | 1137.9–1151.1 | 1136.1–1152.9 | HP2 100%, ADL 28% |
| DPH2 Y | 1144.50 | 1137.2–1151.8 | Off, per A in vacuum | 1137.9–1151.1 | 1135.9–1153.1 | HP2 100%, ADL 28% |
| TL22 | 1166.35 | 1153.6–1179.1 | Material response to current | 1159.0–1174.4 | 1122.7–1127.6, 1147.4–1153.5, 1154.7–1228.5 | ADL 13% |
| DP22 X | 1186.35 | 1179.1–1193.6 | Off, per A in vacuum | 1178.6–1194.1 | 1172.8–1199.9 | TL22 30%, TL21 15% |
| DP22 Y | 1186.35 | 1179.1–1193.6 | Off, per A in vacuum | 1178.5–1194.2 | 1172.5–1200.2 | TL22 30%, TL21 15% |
| HPC X | 1200.85 | 1193.6–1208.1 | Off, per A in vacuum | 1191.9–1209.8 | 1184.0–1217.7 | TL22 23%, TL21 22% |
| HPC Y | 1200.85 | 1193.6–1208.1 | Off, per A in vacuum | 1191.9–1209.8 | 1184.0–1217.7 | TL22 23%, TL21 22% |
| QPC X | 1215.35 | 1208.1–1222.6 | Off, per A in vacuum | 1205.3–1225.4 | 1196.3–1234.4 | TL21 34%, TL22 15% |
| QPC Y | 1215.35 | 1208.1–1222.6 | Off, per A in vacuum | 1205.3–1225.4 | 1196.3–1234.4 | TL21 34%, TL22 15% |
| DP21 X | 1229.90 | 1222.6–1237.2 | Off, per A in vacuum | 1222.1–1237.7 | 1216.4–1243.4 | TL21 44% |
| DP21 Y | 1229.90 | 1222.6–1237.2 | Off, per A in vacuum | 1222.0–1237.8 | 1216.0–1243.8 | TL21 44% |
| TL21 | 1249.95 | 1237.2–1262.7 | Material response to current | 1239.1–1285.4 | 1180.0–1289.1 | — |
| QPH1 X | 1284.50 | 1277.2–1291.8 | Off, per A in vacuum | 1274.5–1294.5 | 1265.4–1303.6 | TL21 60%, HP1 48%, HPol Y 41%, HPol X 24% |
| QPH1 Y | 1284.50 | 1277.2–1291.8 | Off, per A in vacuum | 1274.5–1294.5 | 1265.4–1303.6 | TL21 60%, HP1 48%, HPol Y 41%, HPol X 24% |
| HP1 | 1300.85 | 1285.8–1315.8 | Material response to current | 1284.6–1317.2 | 1281.9–1320.0 | HPol Y 85%, HPol X 51% |
| DPH1 X | 1300.85 | 1293.6–1308.1 | Off, per A in vacuum | 1294.2–1307.5 | 1292.5–1309.3 | HP1 100%, HPol Y 85%, HPol X 51% |
| DPH1 Y | 1300.85 | 1293.6–1308.1 | Off, per A in vacuum | 1294.2–1307.5 | 1292.2–1309.5 | HP1 100%, HPol Y 85%, HPol X 51% |
| HPol X | 1317.20 | 1309.9–1324.5 | Material response to current | 1299.3–1328.3 | 1283.0–1335.6 | HP1 50% |
| HPol Y | 1317.20 | 1309.9–1324.5 | Material response to current | 1285.0–1325.9 | 1282.2–1334.1 | HP1 50% |
| QPol X | 1331.75 | 1324.5–1339.0 | Off, per A in vacuum | 1321.7–1341.8 | 1312.7–1350.8 | HPol X 25%, HPol Y 18% |
| QPol Y | 1331.75 | 1324.5–1339.0 | Off, per A in vacuum | 1321.7–1341.8 | 1312.7–1350.8 | HPol X 25%, HPol Y 18% |
| DP11 X | 1346.30 | 1339.0–1353.6 | Off, per A in vacuum | 1338.5–1354.1 | 1332.8–1359.8 | TL12 26% |
| DP11 Y | 1346.30 | 1339.0–1353.6 | Off, per A in vacuum | 1338.4–1354.2 | 1332.4–1360.2 | TL12 26% |
| TL12 | 1366.30 | 1353.6–1379.0 | Vacuum winding field | 1356.3–1410.5 | 1331.9–1458.5 | — |
| Upper scan X | 1387.50 | 1380.0–1420.0 | Off, per A in vacuum | 1379.2–1395.8 | 1373.0–1402.0 | TL12 94%, Objective 45% |
| Upper scan Y | 1387.50 | 1380.0–1420.0 | Off, per A in vacuum | 1378.1–1396.9 | 1370.5–1404.5 | TL12 94%, Objective 45% |
| Lower scan X | 1412.50 | 1380.0–1420.0 | Off, per A in vacuum | 1404.2–1420.8 | 1398.0–1427.0 | TL12 47%, Objective 45% |
| Lower scan Y | 1412.50 | 1380.0–1420.0 | Off, per A in vacuum | 1403.1–1421.9 | 1395.5–1429.5 | TL12 47%, Objective 45% |
| Condenser stigmator X | 1451.50 | 1436.5–1466.5 | Off, per A in vacuum | 1441.1–1461.9 | 1433.3–1469.7 | Objective 63%, TL12 12% |
| Condenser stigmator Y | 1451.50 | 1436.5–1466.5 | Off, per A in vacuum | 1441.1–1461.9 | 1433.3–1469.7 | Objective 63%, TL12 12% |
| TL11 / Mini | 1522.50 | 1512.5–1532.5 | Vacuum winding field | 1517.5–1546.6 | 1494.6–1570.7 | Objective 83% |
| Objective | 1599.20 | 1389.0–1809.4 | Material field | 1374.9–1386.9, 1419.3–1535.1 | 1367.2–1537.6 | — |

The paired saddle stations list their shared assembly envelope. The center marks each winding station. Unpowered channels show the response to 1 A in the individual X or Y vacuum winding with physical turns. For HP1, HP2, HPol X/Y and TL21/TL22 the percent bands use the local current response of the nonlinear material model. The static round-lens fields for C1–C3, ADL, TL12 and Mini use the signed vacuum winding groups. The objective uses its single material-field supply. [Profiles and solver checks](evidence/field-footprints.json)

## Passive and optical reference planes

| Station | z (mm) | Excited channels above 10% of own peak |
| --- | ---: | --- |
| C2 aperture | 765.000 | — |
| C3 aperture | 1010.000 | C3 74% |
| Virtual scan pivot | 1496.722 | TL11 / Mini 11%, Objective 68% |
| Two-lens front focus | 1490.375 | Objective 67% |
| Specimen entrance | 1599.200 | — |

DP12 uses the same four upper/lower AC winding channels. Its logical offset has no additional field interval. The physical condenser apertures intercept particles at the foil. The virtual pivot and combined-field front focal plane are optical reference planes. The sleeve, return plates and pole pieces modify the response of their magnetic circuit.

## Objective field near and behind the specimen

The isolated objective solution extends beyond the incident integration domain. The entries below give its signed axial field at the corresponding hardware centers. The post-specimen imaging and projection systems require their own operating state.

| Station | z (mm) | Objective Bz (T) |
| --- | ---: | ---: |
| Specimen entrance | 1599.200 | -0.00710 |
| Physical objective aperture | 1600.700 | -0.00638 |
| Image / diffraction upper | 1674.400 | -0.07891 |
| Image / diffraction lower | 1699.400 | -0.06218 |
| Objective stigmator | 1746.900 | -0.05691 |

Over the full column module the objective exceeds 50% of its peak at 1374.9–1386.9, 1419.3–1535.1, 1663.3–1779.3 mm and 10% at 1367.3–1537.6, 1660.8–1809.4 mm. The incident-only bands above use the incident maximum.

The installed outgoing windings have the following unit-current vacuum responses over the specimen-to-module-end interval. A separate imaging state is needed to assign their actual excitation.

| Outgoing winding | Center | ≥50% | ≥10% |
| --- | ---: | --- | --- |
| Image / diffraction upper X | 1674.40 | 1666.1–1682.7 | 1659.9–1688.9 |
| Image / diffraction upper Y | 1674.40 | 1665.0–1683.8 | 1657.4–1691.4 |
| Image / diffraction lower X | 1699.40 | 1691.1–1707.7 | 1684.9–1713.9 |
| Image / diffraction lower Y | 1699.40 | 1690.0–1708.8 | 1682.4–1716.4 |
| Objective stigmator X | 1746.90 | 1736.5–1757.3 | 1728.7–1765.1 |
| Objective stigmator Y | 1746.90 | 1736.5–1757.3 | 1728.7–1765.1 |

The physical objective aperture is downstream of the specimen in this column. Its location differs from the illumination-side coma-free plane. Descan shares the physical image/diffraction deflector in this configuration. The source upstream of the frozen gun exit, energy filter, post-column detector and optional image correctors have no working-point field attribution in this incident trace.
