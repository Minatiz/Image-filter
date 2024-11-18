#include <stdio.h>
#include <stdlib.h>
#include "lib/bmp.h"
#include <math.h>
#include <string.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define GP(X, Y) GetPixel(img, width, height, (X), (Y))

/**
 * @brief Convert an image to grayscale
 *
 * @param img
 * @param width
 * @param height
 */
void ImageToGrayscale(RGB *img, const int width, const int height)
{
    for (int i = 0; i < width * height; i++)
    {
        char grayscale = img[i].red * 0.3 + img[i].green * 0.59 + img[i].blue * 0.11;
        img[i].red = grayscale;
        img[i].green = grayscale;
        img[i].blue = grayscale;
    }
}

/**
 * @brief Return a pixel no matter the coordinate
 *
 * @param img
 * @param width
 * @param height
 * @param x
 * @param y
 * @return RGB
 */
RGB GetPixel(RGB *img, const int width, const int height, const int x, const int y)
{
    if (x < 0 || y < 0 || x >= width || y >= height)
    {
        RGB pixel;
        pixel.red = 0;
        pixel.green = 0;
        pixel.blue = 0;
        return (pixel);
    }
    return (img[x + y * width]);
}

void ApplySobel(RGB *img, const int width, const int height)
{
    // Sobel convolution operator. For Kernel multiplication 3x3
    int Gx[3][3] = {{-1, 0, 1},
                    {-2, 0, 2},
                    {-1, 0, 1}};

    int Gy[3][3] = {{1, 2, 1},
                    {0, 0, 0},
                    {-1, -2, -1}};

    // Allocating memory for storing values in this temp image.
    RGB *tempsobel = malloc(sizeof(RGB) * width * height);
    if (tempsobel == NULL)
    {
        fprintf(stderr, "No memory allocated for tempsobel\n");
        exit(1);
    }

    // Applying first greyscale to the image. For further processing.
    ImageToGrayscale(img, width, height);

    // Looping through image width
    for (int x = 0; x < width; x++)
    {

        // Looping through image height
        for (int y = 0; y < height; y++)
        {
            double sumX = 0;
            double sumY = 0;

            // Adding Gx, Gy kernels to the red channel. Since in greyscaled image all channels RGB has same X-Y values.
            // Gx kernel multiplied with sub-image
            sumX += GP(x - 1, y + 1).red * Gx[0][0]; // Top left
            sumX += GP(x + 0, y + 1).red * Gx[0][1]; // Top center
            sumX += GP(x + 1, y + 1).red * Gx[0][2]; // Top right
            sumX += GP(x - 1, y + 0).red * Gx[1][0]; // Mid left
            sumX += GP(x + 0, y + 0).red * Gx[1][1]; // Current pixel
            sumX += GP(x + 1, y + 0).red * Gx[1][2]; // Mid right
            sumX += GP(x - 1, y - 1).red * Gx[2][0]; // Low left
            sumX += GP(x + 0, y - 1).red * Gx[2][1]; // Low center
            sumX += GP(x + 1, y - 1).red * Gx[2][2]; // Low right

            // Gy kernel multiplied with sub-image.
            sumY += GP(x - 1, y + 1).red * Gy[0][0]; // Top left
            sumY += GP(x + 0, y + 1).red * Gy[0][1]; // Top center
            sumY += GP(x + 1, y + 1).red * Gy[0][2]; // Top right
            sumY += GP(x - 1, y + 0).red * Gy[1][0]; // Mid left
            sumY += GP(x + 0, y + 0).red * Gy[1][1]; // Current pixel
            sumY += GP(x + 1, y + 0).red * Gy[1][2]; // Mid right
            sumY += GP(x - 1, y - 1).red * Gy[2][0]; // Low left
            sumY += GP(x + 0, y - 1).red * Gy[2][1]; // Low center
            sumY += GP(x + 1, y - 1).red * Gy[2][2]; // Low right

            // The function gradient magnitude
            double G = sqrt(sumX * sumX + sumY * sumY);

            // printf(" G: %f, sumX: %f, sumY: %f, x: %d y: %d \n", G, sumX, sumY, x, y);

            // When the gredient magnitude overshoots, setting it as high as possible.
            if (G > 255)
            {
                G = 255;
            }

            // Adding the gradient magnitude value to all channels and storing it in temp image.
            tempsobel[x + y * width].red = G;
            tempsobel[x + y * width].green = G;
            tempsobel[x + y * width].blue = G;
        }
    }

    // Copying the values stored in the temp image to the image to output.
    memcpy(img, tempsobel, sizeof(RGB) * width * height);

    // Free memory
    free(tempsobel);
}

// Code below follows psuedocode from here: https://en.wikipedia.org/wiki/Box_blur#Implementation
void ApplyBoxBlur(RGB *img, const int width, const int height)
{

    // Allocating memory for storing values in this temp image.
    RGB *tempblur = malloc(sizeof(RGB) * width * height);
    if (tempblur == NULL)
    {
        fprintf(stderr, "No memory allocated for tempblur\n");
        exit(1);
    }

    // Looping through image width
    for (int x = 0; x < width; x++)
    {
        // Looping through image height
        for (int y = 0; y < height; y++)
        {
            // Sum variables for each color channel
            double sumRed = 0, sumGreen = 0, sumBlue = 0;

            // Sum of all red in the 3x3 kernel
            sumRed += GP(x - 1, y + 1).red; // Top left
            sumRed += GP(x + 0, y + 1).red; // Top center
            sumRed += GP(x + 1, y + 1).red; // Top right
            sumRed += GP(x - 1, y + 0).red; // Mid left
            sumRed += GP(x + 0, y + 0).red; // Current pixel
            sumRed += GP(x + 1, y + 0).red; // Mid right
            sumRed += GP(x - 1, y - 1).red; // Low left
            sumRed += GP(x + 0, y - 1).red; // Low center
            sumRed += GP(x + 1, y - 1).red; // Low right

            // Sum of all green in the 3x3 kernel
            sumGreen += GP(x - 1, y + 1).green; // Top left
            sumGreen += GP(x + 0, y + 1).green; // Top center
            sumGreen += GP(x + 1, y + 1).green; // Top right
            sumGreen += GP(x - 1, y + 0).green; // Mid left
            sumGreen += GP(x + 0, y + 0).green; // Current pixel
            sumGreen += GP(x + 1, y + 0).green; // Mid right
            sumGreen += GP(x - 1, y - 1).green; // Low left
            sumGreen += GP(x + 0, y - 1).green; // Low center
            sumGreen += GP(x + 1, y - 1).green; // Low right

            // Sum of all blue in the 3x3 kernel
            sumBlue += GP(x - 1, y + 1).blue; // Top left
            sumBlue += GP(x + 0, y + 1).blue; // Top center
            sumBlue += GP(x + 1, y + 1).blue; // Top right
            sumBlue += GP(x - 1, y + 0).blue; // Mid left
            sumBlue += GP(x + 0, y + 0).blue; // Current pixel
            sumBlue += GP(x + 1, y + 0).blue; // Mid right
            sumBlue += GP(x - 1, y - 1).blue; // Low left
            sumBlue += GP(x + 0, y - 1).blue; // Low center
            sumBlue += GP(x + 1, y - 1).blue; // Low right

            // Dividing the sum with 9, and adding it to the temp image.
            tempblur[x + y * width].red = sumRed / 9;
            tempblur[x + y * width].green = sumGreen / 9;
            tempblur[x + y * width].blue = sumBlue / 9;
        }
    }

    // Copying the values stored in the temp image to the image to output.
    memcpy(img, tempblur, sizeof(RGB) * width * height);

    // Free memory
    free(tempblur);
}

int main()
{
    // // const char *marguerite = "marguerite.bmp"; // small 
    // // const char *marguerite = "medium.bmp";
    // const char *marguerite = "large.bmp";   // large 
    const char *marguerite = "xtralarge.bmp";   // xtra large 


    const char *outsobel = "sobel.bmp";
    const char *outbblur = "boxblur.bmp";

    // Get image dimensions
    int width, height;
    GetSize(marguerite, &width, &height);

    // Init memory
    RGB *sobel = malloc(sizeof(RGB) * width * height *2);
    RGB *bblur = malloc(sizeof(RGB) * width * height *2);

    // Load images
    LoadRegion(marguerite, 0, 0, width, height, sobel);
    LoadRegion(marguerite, 0, 0, width, height, bblur);

    // Apply filters
    ApplySobel(sobel, width, height);
    ApplyBoxBlur(bblur, width, height);

    // Save images
    CreateBMP(outsobel, width, height);
    WriteRegion(outsobel, 0, 0, width, height, sobel);
    CreateBMP(outbblur, width, height);
    WriteRegion(outbblur, 0, 0, width, height, bblur);

    // Free memory
    free(sobel);
    free(bblur);

    return (0);
}
