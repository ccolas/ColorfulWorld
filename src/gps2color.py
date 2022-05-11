import numpy as np
import matplotlib.pyplot as plt
import colorspace
from colorspace.colorlib import HCL

example_coordinates = dict(aukland=dict(coordinates=(-36.848461, 174.763336, 196)),
                           boston=dict(coordinates=(42.361145, -71.057083, 49)),
                           cuzco=dict(coordinates=(-13.5226, -71.9673, 3399)),
                           everest=dict(coordinates=(27.986065, 86.922623, 8849)),
                           johannesburg=dict(coordinates=(-26.195246, 28.034088, 1753)),
                           kaboul=dict(coordinates=(34.543896, 69.160652, 1791)),
                           lagos=dict(coordinates=(6.465422, 3.406448, 41)),
                           mariana_trench=dict(coordinates=(11.3500, 142.2000, -10994)),
                           pahoa=dict(coordinates=(19.501225, -154.952881, 200)),
                           paris=dict(coordinates=(48.856614, 2.3522219, 42)),
                           tokyo=dict(coordinates=(35.652832, 139.839478, 40)),
                           zurich=dict(coordinates=(47.3667, 8.5500, 408)),
                           )

range_latitude = (-90, 90)
range_longitude = (-180, 180)
range_altitude = (-431, 8849)

range_hcl_lum = (0, 100)
range_hcl_hue = (0, 360)
range_hcl_chroma = (0, 100)

def sigmoid(x, beta, shift):
    return 1 / (1 + np.exp(-beta * (x + shift)))

def rescale_gps_coordinates(gps_coordinates):
    # linear mapping for now
    latitude, longitude, altitude = gps_coordinates.transpose()
    # latitude = (latitude - range_latitude[0]) / (range_latitude[1] - range_latitude[0])
    longitude = (longitude - range_longitude[0]) / (range_longitude[1] - range_longitude[0])
    # altitude = (altitude - range_altitude[0]) / (range_altitude[1] - range_altitude[0])
    altitude = sigmoid(altitude, beta=1/300, shift=0)
    return (latitude, longitude, altitude)

def get_hcl_from_gps(scaled_gps_coordinates):
    latitude, longitude, altitude = scaled_gps_coordinates
    hue = longitude * (range_hcl_hue[1] - range_hcl_hue[0]) + range_hcl_hue[0]
    chroma = altitude * (range_hcl_chroma[1] - range_hcl_chroma[0]) + range_hcl_chroma[0]
    luminance_scaled = (np.sin(latitude / 180 * np.pi) + 1) / 2
    luminance = luminance_scaled * (range_hcl_lum[1] - range_hcl_lum[0]) + range_hcl_lum[0]
    return (hue, chroma, luminance)


def gps2color(gps_coordinates):
    rescaled_gps_coordinates = rescale_gps_coordinates(gps_coordinates)
    hue, chroma, luminance = get_hcl_from_gps(rescaled_gps_coordinates)

    color = HCL(hue, chroma, luminance)
    print(hue, chroma, luminance)
    color.to('RGB', fixup=True)
    color.swatchplot()



if __name__ == '__main__':
    coordinates = np.array([example_coordinates[c]['coordinates'] for c in sorted(example_coordinates.keys())])
    gps2color(coordinates)
