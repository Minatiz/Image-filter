#include <stdio.h>
#include <stdlib.h>
#include "lib/bmp.h"
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define GP(X, Y) GetPixel(img, width, height, (X), (Y))
#define WORK_TAG 0
#define DONE_TAG 1
#define NEIGH_COM 2

void ImageToGrayscale(RGB *img, const int width, const int height)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < width * height; i++)
    {
        // // printf("threads: %d, of : %d\n", omp_get_thread_num(), omp_get_num_threads());

        char grayscale = img[i].red * 0.3 + img[i].green * 0.59 + img[i].blue * 0.11;
        img[i].red = grayscale;
        img[i].green = grayscale;
        img[i].blue = grayscale;
    }
}

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

// Parallelizing nested loop
#pragma omp parallel for schedule(static)
    // Looping through image width
    for (int x = 0; x < width; x++)
    {

        // Looping through image height
        for (int y = 0; y < height; y++)
        {
            // // printf("threads: %d, of : %d\n", omp_get_thread_num(), omp_get_num_threads());
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
                G = 255;

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

// Parallelizing nested loop
#pragma omp parallel for schedule(static)
    // Looping through image width
    for (int x = 0; x < width; x++)
    {
        // Looping through image height
        for (int y = 0; y < height; y++)
        {
            // // printf("threads: %d, of : %d\n", omp_get_thread_num(), omp_get_num_threads());
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

// Debugging tool, printing out a area of rgb array.
void print_array_x(RGB *img, int x_start, int y_start, int x_end, int y_end, int width)
{
    for (int x = x_start; x < x_end; x++)
    {
        for (int y = y_start; y < y_end; y++)
        {
            RGB pixel = img[x + y * width];
            printf("Pixel(x,y): (%d,%d) R: %d, G: %d, B: %d\n", x, y, pixel.red, pixel.green, pixel.blue);
        }
    }
}

int main()
{

    // Intializing MPI
    int world_rank, world_size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlen;
    MPI_Get_processor_name(name, &resultlen);
    MPI_Status status;
    // Amount workers
    int worker_size = world_size - 1;

    // const char *marguerite = "marguerite.bmp"; // small image
    // // const char *marguerite = "medium.bmp";  // medium image
    // // const char *marguerite = "large.bmp";   // large 
    const char *marguerite = "xtralarge.bmp";   // large 

    
    // Get image dimensions. 
    int width, height;
    if(!world_rank)
        GetSize(marguerite, &width, &height);


    // Broadcasting the width and height to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // All processes need to know this.
    // Calculating tile height that each worker receives
    int tile_height_size = height / worker_size;
    // Remainder goes to last process.
    int remainder_height = height % worker_size;

    // Master process delegate work to workers, and receive in the end the processed tile and write to output the image.
    if (!world_rank)
    {

        const char *outsobel = "sobel.bmp";
        const char *outbblur = "boxblur.bmp";

        // Init memory for the images.
        RGB *sobel = malloc(sizeof(RGB) * width * (height));
        RGB *bblur = malloc(sizeof(RGB) * width * (height));
        if (sobel == NULL || bblur == NULL)
        {
            fprintf(stderr, "No memory allocated for sorbel img or box blur img\n");
            return 1;
        }

        // Only master reads whole image.
        LoadRegion(marguerite, 0, 0, width, height, sobel);
        LoadRegion(marguerite, 0, 0, width, height, bblur);

        int start_height = 0;

        // Delegating work to worker. Sending them their work tile with the extra rows.
        for (int worker_rank = 1; worker_rank < world_size; worker_rank++)
        {
            // Calculating end_height range.
            int end_height = tile_height_size;

            // Adding height remainder to the last tile.
            if (worker_rank == worker_size)
                end_height += remainder_height;

            if (worker_rank == 1)
                start_height -= 1;

            // printf("height_start: %d, height_end: %d,  bytes: %lu \n", rows, start_height, end_height,  end_height * width * sizeof(RGB));
            // print_array_x(&sobel[width * start_height], 0, 0, width, end_height, width);


            // Sending the work tile without the extra tiles!
            MPI_Send(&sobel[(start_height)*width], end_height * width * sizeof(RGB), MPI_BYTE, worker_rank, WORK_TAG, MPI_COMM_WORLD);
            // incrementing the start height with end_height
            start_height += end_height;
        }

        start_height = 0;

        // Gathering all tiles from worker
        for (int worker_rank = 1; worker_rank < world_size; worker_rank++)
        {
            // Calculating end_height range.
            int end_height_received = tile_height_size;

            // Adding width and height remainder to the last tile.
            if (worker_rank == worker_size)
                end_height_received += remainder_height;

            // Receive the processed tiles with the filter applied, without the extra rows.
            MPI_Recv(&sobel[start_height * width], end_height_received * width * sizeof(RGB), MPI_BYTE, worker_rank, DONE_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&bblur[start_height * width], end_height_received * width * sizeof(RGB), MPI_BYTE, worker_rank, DONE_TAG, MPI_COMM_WORLD, &status);
            printf("Master received results from worker %d.\n", worker_rank);
            // // print_array_x(&sobel[width * start_height], 0, 0, width, end_height, width);

            // incrementing the start height with end_height
            start_height += end_height_received;
        }

        // Master creates the output BMP and writes the processsed tiles.
        CreateBMP(outsobel, width, height);
        WriteRegion(outsobel, 0, 0, width, height, sobel);
        CreateBMP(outbblur, width, height);
        WriteRegion(outbblur, 0, 0, width, height, bblur);

        // Free memory
        free(sobel);
        free(bblur);
    }

    // Worker processes. Gets the tile and applies the filtering function and worker communicates between each other.
    else
    {
        // Calculating tile height which is row for worker.
        int row = tile_height_size; 
        if (world_rank == worker_size) // last worker
            row += remainder_height; 

        // Receiving the row amount from master.
        // // printf("rows: %d\n", row);

        // Allocating memory for the tile receiving from worker. And 2 extra rows for memcpy below when neighbour com.
        RGB *sorbel_tile = malloc(sizeof(RGB) * width * (row + 2));
        RGB *bblur_tile = malloc(sizeof(RGB) * width * (row + 2));
        if (sorbel_tile == NULL || bblur_tile == NULL)
        {
            fprintf(stderr, "No memory allocated for sorbel tile or box blur tile\n");
            return 1;
        }

        // Receiving the work tiles.
        MPI_Recv(&sorbel_tile[width], sizeof(RGB) * width * row, MPI_BYTE, 0, WORK_TAG, MPI_COMM_WORLD, &status);
        // Copying same tile information to the bblur tile. Fixes issue on cluster nodes when trying larger images. 
        memcpy(&bblur_tile[width], &sorbel_tile[width], sizeof(RGB) * width * row); 

        // Allocating memory for the top and bot boundary rows.
        RGB *topsob = malloc(sizeof(RGB) * width * row);
        RGB *botsob = malloc(sizeof(RGB) * width * row);
        RGB *topbb = malloc(sizeof(RGB) * width * row);
        RGB *botbb = malloc(sizeof(RGB) * width * row);
        if (topsob == NULL || botsob == NULL || topbb == NULL || botbb == NULL)
        {
            fprintf(stderr, "No memory allocated for top or bot \n");
            return 1;
        }

        // Worker process neighbour. World_rank is this worker and he has neighbour below and above
        int neighbour_below = world_rank - 1;
        int neighbour_above = world_rank + 1;

        // Get the missing rows from different neighbors. Getting missing rows for filling the missing pixels by communicating with workers only.
        // This will always fetch.
        if (world_rank > 1)
        {
            printf("1: My rank: %d, neighbour_above: %d, neighbour_below: %d \n", world_rank, neighbour_above, neighbour_below);
            // Sending and receiving from the neighbour below my rank it's rows. Gives me top boundary.
            MPI_Sendrecv(&sorbel_tile[width], width * sizeof(RGB), MPI_BYTE, neighbour_below, NEIGH_COM, topsob, width * sizeof(RGB), MPI_BYTE, neighbour_below, NEIGH_COM, MPI_COMM_WORLD, &status);
            MPI_Sendrecv(&bblur_tile[width], width * sizeof(RGB), MPI_BYTE, neighbour_below, NEIGH_COM, topbb, width * sizeof(RGB), MPI_BYTE, neighbour_below, NEIGH_COM, MPI_COMM_WORLD, &status);
            // // print_array_x(topsob, 0, 0, width, 1, width);

            // // Copying the received top boundary row to the worker tile. This part was missed why i was seeing white lines.
            memcpy(&sorbel_tile[0], topsob, sizeof(RGB) * width);
            memcpy(&bblur_tile[0], topbb, sizeof(RGB) * width);
        }
        if (world_rank < worker_size)
        {
            printf("2: My rank: %d, neighbour_above: %d, neighbour_below: %d \n", world_rank, neighbour_above, neighbour_below);
            // Sending and receiving from the neighbour above my rank it's rows. Gives me bot boundary.
            MPI_Sendrecv(&sorbel_tile[row * width], width * sizeof(RGB), MPI_BYTE, neighbour_above, NEIGH_COM, botsob, width * sizeof(RGB), MPI_BYTE, neighbour_above, NEIGH_COM, MPI_COMM_WORLD, &status);
            MPI_Sendrecv(&bblur_tile[row * width], width * sizeof(RGB), MPI_BYTE, neighbour_above, NEIGH_COM, botbb, width * sizeof(RGB), MPI_BYTE, neighbour_above, NEIGH_COM, MPI_COMM_WORLD, &status);
            // // print_array_x(botsob, 0, 0, width, 1, width);
            // Copying the received bot boundary row to the worker tile. This part was missed why i was seeing white lines.
            memcpy(&sorbel_tile[(row + 1) * width], botsob, sizeof(RGB) * width);
            memcpy(&bblur_tile[(row + 1) * width], botbb, sizeof(RGB) * width);
        }

        // Worker applies the filters to their respective tiles plus the fetched extra tiles from workers.
        // Top and bot extra rows.
        ApplySobel(sorbel_tile, width, row + 2);
        ApplyBoxBlur(bblur_tile, width, row + 2);
        printf("Worker %d sending Sobel and Box Blur results back to master.\n", world_rank);

        // After worker applied the filter on the tiles, send it back to master for writing it.
        MPI_Send(&sorbel_tile[width], (row)*width * sizeof(RGB), MPI_BYTE, 0, DONE_TAG, MPI_COMM_WORLD);
        MPI_Send(&bblur_tile[width], (row)*width * sizeof(RGB), MPI_BYTE, 0, DONE_TAG, MPI_COMM_WORLD);

        // Free memory
        free(sorbel_tile);
        free(bblur_tile);
        free(topsob);
        free(botsob);
        free(topbb);
        free(botbb);
    }

    // Cleans and shuts down MPI.
    MPI_Finalize();

    return 0;
}