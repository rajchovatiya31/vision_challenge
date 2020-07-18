import cv2
import numpy as np
from math import pi, cos, sin
from collections import defaultdict


def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])
    # multiply the angles by two and find coordinates of that angle
    points = np.array([[np.cos(2 * angle), np.sin(2 * angle)]
                       for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(points, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Input: Two lines in hough space in form [[rho, theta]]
    Returns: closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]


def segmented_intersections(lines):
    """
    Finds the intersections between groups of lines.

    lines: array of hough lines
    return: Array of intersection points
    """

    intersections = []
    # loop over all lines except last
    for i, line in enumerate(lines[:-1]):
        # loop form next to all
        for next_line in lines[i + 1:]:
            for line1 in line:
                for line2 in next_line:
                    intersections.append(intersection(line1, line2))

    return intersections


def get_polygon_points(segmented_points, intersection_points):
    """
    Description: Get outer most point of the square from the intersection points

    segmented_points: list of segmented lines
    intersection_points: list of intersection points in format [[x1, y1], [x2,y2], ...]
    """
    poly_points = []
    if np.squeeze(segmented_points)[0][0][1] != 0:
        x_min, y_min = np.squeeze(np.argmin(intersection_points, axis=0))
        x_max, y_max = np.squeeze(np.argmax(intersection_points, axis=0))
        bound_list = [x_min, y_min, x_max, y_max]
        for i in bound_list:
            poly_points.append([intersection_points[i][0], intersection_points[i][1]])
    else:
        x_min, y_min = np.squeeze(np.min(intersection_points, axis=0))
        x_max, y_max = np.squeeze(np.max(intersection_points, axis=0))
        poly_points = [[x_min, y_min], [x_max, y_min], [y_max, y_max], [x_min, y_max]]

    poly_points = np.array(poly_points).reshape((-1, 1, 2)).astype(np.int32)

    return poly_points


class EdgeDetector:

    def __init__(self, image):
        self.image = image
        self.output_image = image.copy()

    def set_image(self, img):
        """
        Description: Image setter.
        """
        self.image = img
        self.output_image = img

    def image_preprocessing(self, img, **kwargs):
        """
        Description: This function is aiming to preprocess the image such a way that the raw image will be passed
                     and using other parameters it will return denoised image. After preprocessing image will be
                     ready for edge detection.

        img: Raw image pointer.
        gaussian_ksize: Kernel size for gaussian blur filter. Kernel size is intended to be a square size of
                        (gaussian_ksize x gaussian_ksize)
        sigma_x:
        threshold: Threshold used for OTSU thresholding.
        max_value: Maximum value to use with the THRESH_BINARY.
        filter_strength: Filter strength for mean denoising.
        templet_window_size: Size of template patch that is used to compute weights
        search_window_size: Size of window that is used to compute weighted average for given pixel
        Return: Preprocessed image.
        """
        # exta parameters could be set by kwargs
        gaussian_ksize = kwargs.get('gaussian_ksize', 5)
        sigma_x = kwargs.get('sigma_x', 0)
        threshold = kwargs.get('threshold', 0)
        max_value = kwargs.get('max_value', 255)
        flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
        filter_strength = kwargs.get('filter_strength', 30)
        templet_window_size = kwargs.get('templet_window_size', 7)
        search_window_size = kwargs.get('search_window_size', 21)

        # Preprocessing
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to gray scale
        gray_image_blur = cv2.GaussianBlur(gray_image, (gaussian_ksize, gaussian_ksize), sigma_x)  # Gaussian blur
        _, binary_img = cv2.threshold(gray_image_blur, threshold, max_value, flags)  # Convert to binary mask
        binary_dst = cv2.fastNlMeansDenoising(binary_img, None, filter_strength, templet_window_size,
                                              search_window_size) # Denoised binary mask

        return binary_dst

    def edge_detection(self, binary_img, **kwargs):
        """
        Description: Function to perform edge detection on the preprocessed_img image.

        preprocessed_img: Preprocessed image from preprocessing step.
        canny_threshold: Lower threshold for hysteresis procedure.
        ratio: Ration to adjust lowThreshold parameter and pass it to threshold2 parameter of Canny.
               (threshold2= lowThreshold * ratio)
        kernel_size: apertureSize  parameter of Canny.
        rho: Parameter of HoughLines - distance resolution of accumulator in pixel.
        theta: Parameter of HoughLines - angle resolution of accumulator in radians.
        hough_threshold: Parameter of HoughLines - accumulator threshold.
        Return: Vector of lines found by hough transformation.
        """
        # exta parameters could be set by kwargs
        canny_threshold = kwargs.get('canny_threshold', 50)
        ratio = kwargs.get('ratio', 4)
        kernel_size = kwargs.get('kernel_size', 3)
        rho = kwargs.get('rho', 1)
        theta = kwargs.get('theta', pi / 180)
        hough_threshold = kwargs.get('hough_threshold', 150)

        # edge detection process
        canny_lines = cv2.Canny(binary_img, canny_threshold, ratio * canny_threshold, None, kernel_size)
        hough_lines = cv2.HoughLines(canny_lines, rho, theta, hough_threshold, None)

        return hough_lines

    def cut_extra_lines(self, poly_points):
        """
        Description: Function to cut extra line from output image detected by hough transformation

        poly_points: Numpy array of polygon points(square outermost points)
        """
        blank_image = np.zeros_like(self.image).astype(np.uint8)
        if len(poly_points) > 0:
            cv2.fillPoly(blank_image, [poly_points], (255, 255, 255)) # make binary mask of rectangle on blank image
            # put rectangle mask on output image
            self.output_image[:, :, 0] = np.where(blank_image[:, :, 0] == 0, 0, self.output_image[:, :, 0])
            self.output_image[:, :, 1] = np.where(blank_image[:, :, 1] == 0, 0, self.output_image[:, :, 1])
            self.output_image[:, :, 2] = np.where(blank_image[:, :, 2] == 0, 0, self.output_image[:, :, 2])
        else:
            pass


    def draw_lines(self, lines, color=(0, 255, 0), line_thickness=3):
        """
        Description: Function to draw lines on image found in edge detection step.

        lines: Vector of lines found in edge detection step.
        line_thickness: Thickness of line needs to be drawn in pixel.
        """
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                x0 = cos(theta) * rho  # convert to cartesian coordinates
                y0 = sin(theta) * rho
                # find end points of line
                pt1 = (int(x0 + 1000 * (-sin(theta))), int(y0 + 1000 * cos(theta)))
                pt2 = (int(x0 - 1000 * (-sin(theta))), int(y0 - 1000 * cos(theta)))
                cv2.line(self.output_image, pt1, pt2, color, line_thickness, cv2.LINE_AA)
        else:
            pass

    def show_image(self, img, window_name="output"):
        """
        Description: Function to display the image.

        img: Image to be displayed.
        window_name: Name of the display window
        """

        # if image is not empty display the image
        if img is not None:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_name, img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            print("Image is empty!")

    def detect_edge(self):
        """
        Description: Function that combines all steps of process
        """
        preprocessed_img = self.image_preprocessing(self.image)
        hough_lines_binary = self.edge_detection(preprocessed_img)
        self.draw_lines(hough_lines_binary)
        segmented_lines = segment_by_angle_kmeans(hough_lines_binary)
        intersections = np.array(segmented_intersections(segmented_lines))
        poly_points = get_polygon_points(segmented_lines, intersections)
        self.cut_extra_lines(poly_points)
        self.show_image(self.output_image)
        cv2.imwrite('../output/output.png', self.output_image)


if __name__ == "__main__":
    image = cv2.imread('../data/Image_4.png')
    edge_detector = EdgeDetector(image)
    edge_detector.detect_edge()




