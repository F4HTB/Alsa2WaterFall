#include "Alsa2WaterFall.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

#include <alsa/asoundlib.h>

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <string.h>

#include <png.h>
#include <signal.h>
#include <math.h>

#include <time.h>
#include <pthread.h>

bool VERBOSE = 0;

/*Audio*/

struct soundInfo
{
	
	char *BUFFER;
	char *DEVICE = (char*)"default";
	unsigned int RATE = 48000;
	int  SAMPLES_PER_TURN = 4096;
	int  CHANEL = 2;
	
	snd_pcm_uframes_t BUFFER_SIZE = 4096 * 2 * 2; /* 2 bytes per sample, 2 channel */
	
	snd_pcm_t *CAPTURE_HANDLE;
	snd_pcm_hw_params_t *HW_PARAMS;
	snd_pcm_format_t FORMAT = SND_PCM_FORMAT_S16_LE;
	
} sound;

void audioInit(void)
{

	int err;

	if ((err = snd_pcm_open (&sound.CAPTURE_HANDLE, sound.DEVICE, SND_PCM_STREAM_CAPTURE, 0)) < 0) {
		fprintf (stderr, "cannot open audio device %s (%s)\n", sound.DEVICE, snd_strerror (err));
		exit (1);
	}
	fprintf(stdout, "audio interface opened %s\n",sound.DEVICE);


	if ((err = snd_pcm_hw_params_malloc (&sound.HW_PARAMS)) < 0) {
		fprintf (stderr, "cannot allocate hardware parameter structure (%s)\n", snd_strerror (err));
		exit (1);
	}
	fprintf(stdout, "hw_params allocated\n");


	if ((err = snd_pcm_hw_params_any (sound.CAPTURE_HANDLE, sound.HW_PARAMS)) < 0) {
		fprintf (stderr, "cannot initialize hardware parameter structure (%s)\n", snd_strerror (err));
		exit (1);
	}
	fprintf(stdout, "hw_params initialized\n");


	if ((err = snd_pcm_hw_params_set_access (sound.CAPTURE_HANDLE, sound.HW_PARAMS, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
		fprintf (stderr, "cannot set access type (%s)\n", snd_strerror (err));
		exit (1);
	}
	fprintf(stdout, "hw_params access setted\n");


	if ((err = snd_pcm_hw_params_set_format (sound.CAPTURE_HANDLE, sound.HW_PARAMS, sound.FORMAT)) < 0) {
		fprintf (stderr, "cannot set sample format (%s)\n", snd_strerror (err));
		exit (1);
	}
	fprintf(stdout, "hw_params format setted %d (2=SND_PCM_FORMAT_S16_LE)\n",sound.FORMAT);
	
	
	if ((err = snd_pcm_hw_params_set_rate_near (sound.CAPTURE_HANDLE, sound.HW_PARAMS, &sound.RATE, 0)) < 0) {
		fprintf (stderr, "cannot set sample rate (%s)\n", snd_strerror (err));
		exit (1);
	}
	fprintf(stdout, "hw_params rate setted %d\n",sound.RATE);

	snd_pcm_uframes_t period_size_nearest_to_a_target = sound.SAMPLES_PER_TURN;
	if ((err = snd_pcm_hw_params_set_period_size_near (sound.CAPTURE_HANDLE, sound.HW_PARAMS, &period_size_nearest_to_a_target, 0)) < 0) {
		fprintf (stderr, "cannot set period size (%s)\n", snd_strerror (err));
		exit (1);
	}
	fprintf(stdout, "hw_params period size setted %d\n",sound.SAMPLES_PER_TURN);


	if ((err = snd_pcm_hw_params_set_channels (sound.CAPTURE_HANDLE, sound.HW_PARAMS, sound.CHANEL)) < 0) {
		fprintf (stderr, "cannot set channel count (%s)\n", snd_strerror (err));
		exit (1);
	}
	fprintf(stdout, "hw_params channels setted %d\n",sound.CHANEL);


	if ((err = snd_pcm_hw_params (sound.CAPTURE_HANDLE, sound.HW_PARAMS)) < 0) {
		fprintf (stderr, "cannot set parameters (%s)\n", snd_strerror (err));
		exit (1);
	}
	fprintf(stdout, "hw_params setted\n");


	snd_pcm_hw_params_free (sound.HW_PARAMS);

	fprintf(stdout, "hw_params freed\n");

	if ((err = snd_pcm_prepare (sound.CAPTURE_HANDLE)) < 0) {
		fprintf (stderr, "cannot prepare audio interface for use (%s)\n", snd_strerror (err));
		exit (1);
	}
	fprintf(stdout, "audio interface prepared\n");

}

int audioRead(void)
{
	snd_pcm_sframes_t rc;
	snd_pcm_drop(sound.CAPTURE_HANDLE);
	snd_pcm_prepare(sound.CAPTURE_HANDLE);
	rc = snd_pcm_readi (sound.CAPTURE_HANDLE, sound.BUFFER, sound.SAMPLES_PER_TURN);
	if (rc == -EPIPE)
	{
		/* EPIPE means overrun */
		snd_pcm_recover(sound.CAPTURE_HANDLE, rc, 0);
	}
	else if (rc < 0)
	{
		fprintf(stderr, "error from read: %s\n", snd_strerror(rc));
	}
	else
	{
		if(VERBOSE)fprintf(stdout, "read: %ld samples\n", rc);
	}
	return rc;
}

void audioDeinit(void)
{
	free(sound.BUFFER);
	fprintf(stdout, "Audio buffer freed\n");
	snd_pcm_drop(sound.CAPTURE_HANDLE);
	snd_pcm_close (sound.CAPTURE_HANDLE);
	fprintf(stdout, "Audio interface closed\n");
}

short int getFrame(char *buffer, int i, short int channel)
{
        return (buffer[(4 * i) + channel] & 0xFF) + ((buffer[(4 * i) + 1 + channel] & 0xFF) << 8);
}

/*End Audio*/

/*FFT*/


struct fftwInfo
{
	
	int INTERVAL = 1;
	int AVG = 1;
	int WRITE_OUTPUT_INTERVAL = 10;
	char *OUTPUT_FILE = (char*)"output.png";
	char *OUTPUT_FILE_TMP = (char*)"output.png.tmp";
	
	int OUTLENGHT;
	fftw_complex *IN;
	fftw_complex *OUT;	
	fftw_plan PLAN;
	double *CURRENTLINE;
	
	float *HANNINGWINDOWS;
	
} fftw;


void fftwInit(void)
{
	fftw.OUTLENGHT = sound.SAMPLES_PER_TURN;
	fftw.IN = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fftw.OUTLENGHT);
	fftw.OUT = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (fftw.OUTLENGHT));	
	fftw.PLAN = fftw_plan_dft_1d(fftw.OUTLENGHT, fftw.IN, fftw.OUT, FFTW_BACKWARD, FFTW_MEASURE);
	fftw_init_threads();
	fftw_plan_with_nthreads(FFTW_NO_NONTHREADED);
	fftw.CURRENTLINE = (double *)malloc(sizeof(double) * fftw.OUTLENGHT);
    memset(fftw.CURRENTLINE, 0, sizeof(double) * fftw.OUTLENGHT);
}


void fftwDeinit(void)
{
	fftw_destroy_plan(fftw.PLAN);
	fftw_free(fftw.IN);
	fftw_free(fftw.OUT);
	free(fftw.CURRENTLINE);
	fftw_cleanup();
	fprintf(stdout, "FFT freed\n");
}

float *hanningInit(int N, short itype = 0)
{
        int half, i, idx, n;
        float *w;

        w = (float*) calloc(N, sizeof(float));
        memset(w, 0, N*sizeof(float));

        if(itype==1) //periodic function
                n = N-1;
        else
                n = N;

        if(n%2==0)
        {
                half = n/2;
                for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
                        w[i] = 0.5 * (1 - cos(2*M_PI*(i+1) / (n+1)));

                idx = half-1;
                for(i=half; i<n; i++) {
                        w[i] = w[idx];
                        idx--;
                }
        }
        else
        {
                half = (n+1)/2;
                for(i=0; i<half; i++) //CALC_HANNING   Calculates Hanning window samples.
                        w[i] = 0.5 * (1 - cos(2*M_PI*(i+1) / (n+1)));

                idx = half-2;
                for(i=half; i<n; i++) {
                        w[i] = w[idx];
                        idx--;
                }
        }

        if(itype==1) //periodic function
        {
                for(i=N-1; i>=1; i--)
                        w[i] = w[i-1];
                w[0] = 0.0;
        }
        return(w);
}
/*End FFT*/

/*bitmap*/
/* A coloured pixel. */

typedef struct
{
    uint8_t red;
    uint8_t green;
    uint8_t blue;
}
pixel_t;

/* A picture. */
    
typedef struct
{
    pixel_t *pixels;
    size_t width;
    size_t height = 14400; //24*60*10
	int TEXT_OFFSET = 80;
	int TIMER_MARKER_INTERVAL = 30;
}
bitmap_t;
bitmap_t waterfall;

/* Given "bitmap", this returns the pixel of bitmap at the point 
   ("x", "y"). */

static pixel_t * pixel_at (bitmap_t * bitmap, int x, int y)
{
    return bitmap->pixels + bitmap->width * y + x;
}

static int save_watterfall_to_file (bitmap_t *bitmap, const char *path, int offset_y = 0)
{
    FILE * fp;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    size_t x, y;
    png_byte ** row_pointers = NULL;
    /* "status" contains the return value of this function. At first
       it is set to a value which means 'failure'. When the routine
       has finished its work, it is set to a value which means
       'success'. */
    int status = -1;
    /* The following number is set by trial and error only. I cannot
       see where it it is documented in the libpng manual.
    */
    int pixel_size = 3;
    int depth = 8;
    
    fp = fopen (path, "wb");
    if (! fp) {
        goto fopen_failed;
    }

    png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        goto png_create_write_struct_failed;
    }
    
    info_ptr = png_create_info_struct (png_ptr);
    if (info_ptr == NULL) {
        goto png_create_info_struct_failed;
    }
    
    /* Set up error handling. */

    if (setjmp (png_jmpbuf (png_ptr))) {
        goto png_failure;
    }
    
    /* Set image attributes. */

    png_set_IHDR (png_ptr,
                  info_ptr,
                  bitmap->width,
                  bitmap->height,
                  depth,
                  PNG_COLOR_TYPE_RGB,
                  PNG_INTERLACE_NONE,
                  PNG_COMPRESSION_TYPE_DEFAULT,
                  PNG_FILTER_TYPE_DEFAULT);
    
    /* Initialize rows of PNG. */

    row_pointers = (png_byte **)png_malloc (png_ptr, bitmap->height * sizeof (png_byte *));
    for (y = 0; y < bitmap->height; y++) {
        png_byte *row = (png_byte *)png_malloc (png_ptr, sizeof (uint8_t) * bitmap->width * pixel_size);
        row_pointers[y] = row;
		size_t actu_y = offset_y+y+1;
		if(actu_y>bitmap->height)actu_y-=bitmap->height;
        for (x = 0; x < bitmap->width; x++) {
            pixel_t * pixel = pixel_at (bitmap, x, actu_y);
            *row++ = pixel->red;
            *row++ = pixel->green;
            *row++ = pixel->blue;
        }
    }
    
    /* Write the image data to "fp". */

    png_init_io (png_ptr, fp);
    png_set_rows (png_ptr, info_ptr, row_pointers);
    png_write_png (png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    /* The routine has successfully written the file, so we set
       "status" to a value which indicates success. */

    status = 0;
    
    for (y = 0; y < bitmap->height; y++) {
        png_free (png_ptr, row_pointers[y]);
    }
    png_free (png_ptr, row_pointers);
    
 png_failure:
 png_create_info_struct_failed:
    png_destroy_write_struct (&png_ptr, &info_ptr);
 png_create_write_struct_failed:
    fclose (fp);
 fopen_failed:
    return status;
}

void bitmapInit(void)
{
	waterfall.width = fftw.OUTLENGHT+waterfall.TEXT_OFFSET;
	waterfall.pixels = (pixel_t *)calloc (waterfall.width * waterfall.height, sizeof (pixel_t));
	memset(waterfall.pixels, 0, sizeof(pixel_t) * waterfall.width * waterfall.height);
}

void bitmapDeinit(void)
{
	free(waterfall.pixels);
}

void draw_time(time_t now,size_t linestart)
{
	
	struct tm *now_tm = localtime(&now);
	short time[5];
	time[0]=now_tm->tm_hour/10;
	time[1]=now_tm->tm_hour-time[0]*10;
	time[2]=10;
	time[3]=now_tm->tm_min/10;
	time[4]=now_tm->tm_min-(time[3]*10);
	
	int actu_pixel_index = (linestart * waterfall.width);
	for (int p = 0; p < waterfall.TEXT_OFFSET; p++)
	{
		waterfall.pixels[actu_pixel_index].red = 0;
		waterfall.pixels[actu_pixel_index].green = 255;
		waterfall.pixels[actu_pixel_index].blue = 0;
		actu_pixel_index++;				
	}
	
	int nline_to_delete = (waterfall.TIMER_MARKER_INTERVAL * 60) / fftw.INTERVAL;
	
	for (int x = 1; x < nline_to_delete; x++){
		actu_pixel_index = (((int)linestart + x) > (int)waterfall.height) ? (x * waterfall.width) : ((linestart + x) * waterfall.width);
		for (int p = 0; p < waterfall.TEXT_OFFSET; p++)
		{
			waterfall.pixels[actu_pixel_index].red = 0;
			waterfall.pixels[actu_pixel_index].green = 0;
			waterfall.pixels[actu_pixel_index].blue = 0;
			actu_pixel_index++;				
		}
	}
	
	
	for(int z=0;z<5;z++){
		int line = linestart + 2;
		for(int i=0;i<8;i++){
			line = (line >= (int)waterfall.height) ? 0 : line;
			actu_pixel_index = line * waterfall.width + z * 10;
			for(int j=0;j<10;j++){
				if(numbers[time[z]][j] & (1 << i)){
					waterfall.pixels[actu_pixel_index].green = 255;
				}
				else{waterfall.pixels[actu_pixel_index].green = 0;}
				if(numbers[time[z]][j+10] & (1 << i)){
					waterfall.pixels[actu_pixel_index+(8 * waterfall.width)].green = 255;
				}
				else{waterfall.pixels[actu_pixel_index+(8 * waterfall.width)].green = 0;}
				actu_pixel_index++;
			}
			line++;
		}
		
	}
}

/*End bitmap*/

/*Other*/
static volatile int keepRunning = 1;
void CtrlCHandler(int dummy __attribute__((unused))) {
    keepRunning = 0;
}

#define USLEEP_MAX (1000000 - 1)
void long_sleep(unsigned long micros)
{
  while(micros > 0)
  {
    const unsigned long chunk = micros > USLEEP_MAX ? USLEEP_MAX : micros;
    usleep(chunk);
    micros -= chunk;
  }
}
char* my_strcat(const char* const s1, const char* const s2)
{
    char *dst = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    if (dst == NULL)
    {
      return NULL;
    }
    sprintf(dst, "%s%s", s1, s2);
    return dst;
}
/*End Other*/


/*MainThreads*/
void *writeBitmap(void *vargp) 
{ 
	int *culine = (int *)vargp; 
	if (save_watterfall_to_file (&waterfall, fftw.OUTPUT_FILE_TMP, *culine)) {
				fprintf (stderr, "Error writing file.\n");
	}
	rename(fftw.OUTPUT_FILE_TMP, fftw.OUTPUT_FILE);
    return NULL; 
} 
/*End MainThreads*/
    
int main (int argc, char *argv[])
{
	signal(SIGINT, CtrlCHandler);

	int c;
	while ((c = getopt (argc, argv, "hd:r:i:a:x:w:m:o:v")) != -1)
			switch (c)
			{
			case 'h':
					printf ("Alsa2WaterFall -d plughw:CARD=PCH,DEV=0 -r 192000 -i 60 -a 10 -x 10 -w 1024 -m 10 -o outputfile.bmp -v\n"); //$dsnoop:lp2220mXRP10,1
					printf ("Alsa2WaterFall -d plughw:CARD=PCH,DEV=0 -r 192000\n");
					printf ("-d device\n-r samplerate\n-i interval in s to output a line\n-a number of fft average for each line\n-x all n intervall write output\n-w wide pixel of output\n-m marker in modulo minute for time marker on left\n-o outputfile in bmp\n");
					printf ("format I/Q f16_le only\n");
					exit(0);
					break;
			case 'd':
					sound.DEVICE = optarg;
					break;
			case 'r':
					sound.RATE = atoi(optarg);
					break;
			case 'i':
					fftw.INTERVAL = atoi(optarg);
					break;
			case 'a':
					fftw.AVG = atoi(optarg);
					break;
			case 'x':
					fftw.WRITE_OUTPUT_INTERVAL = atoi(optarg);
					break;
			case 'w':
					sound.SAMPLES_PER_TURN = atoi(optarg);
					break;
			case 'm':
					waterfall.TIMER_MARKER_INTERVAL = atoi(optarg);
					break;
			case 'o':
					fftw.OUTPUT_FILE = optarg;
					fftw.OUTPUT_FILE_TMP = my_strcat(fftw.OUTPUT_FILE,".tmp");
					break;
			case 'v':
					VERBOSE = 1;
					break;			
			default:
					abort ();
			}
			
	
	if(waterfall.TIMER_MARKER_INTERVAL == 0)waterfall.TEXT_OFFSET=0;
	else if(ceilf(((float)fftw.INTERVAL*10)/60) > waterfall.TIMER_MARKER_INTERVAL){waterfall.TIMER_MARKER_INTERVAL = ceilf(((float)fftw.INTERVAL*10)/60);}
	
	if(VERBOSE){
		printf("Get a line all %ds, this mean minimal marker interval must be set to %.0fm, actualy is set to %dm\n", fftw.INTERVAL, ceilf(((float)fftw.INTERVAL*10)/60),waterfall.TIMER_MARKER_INTERVAL);
		printf("With %d average means read audio and fft all %.2fs\n", fftw.AVG, (double)fftw.INTERVAL/(double)fftw.AVG);
		printf("With %d intervall write output mean output all %ds\n", fftw.WRITE_OUTPUT_INTERVAL, fftw.WRITE_OUTPUT_INTERVAL*fftw.INTERVAL);
		int h = ((waterfall.height*fftw.INTERVAL)/3600); 
		int m = ((waterfall.height*fftw.INTERVAL) -(3600*h))/60;
		int s = ((waterfall.height*fftw.INTERVAL) -(3600*h)-(m*60));		
		printf("total recording time on %ld lines is %d:%d:%d\n", waterfall.height,h,m,s);
	}
	

	sound.BUFFER_SIZE = sound.SAMPLES_PER_TURN * 2 * 2;
	sound.BUFFER = (char *)malloc(sound.BUFFER_SIZE);
	memset(sound.BUFFER, 0, sizeof(char) * sound.BUFFER_SIZE);


	audioInit();
	fftwInit();
	fftw.HANNINGWINDOWS = hanningInit(sound.SAMPLES_PER_TURN);

	bitmapInit();
	
	
	int WRITE_OUTPUT_INTERVAL_counter = fftw.WRITE_OUTPUT_INTERVAL;
	
	time_t now = time( NULL );
	struct tm *now_tm;
	struct timespec tp;
	int hour,min,sec,last_maker_time;
	
	pthread_t writeBitmap_thread_id; 
  
	size_t actu_line_index=0;
	while (keepRunning) { 
	
		bool CURRENTLINE_SET=1;
		for (int a = 0; a < fftw.AVG; a++)
		{
			
			clock_gettime(CLOCK_REALTIME, &tp);
			unsigned long start_stop = tp.tv_sec * 1000000 + tp.tv_nsec/1000;
			
			audioRead();
			
			for (int i = 0; i < (int)sound.SAMPLES_PER_TURN; i++)
			{
					short int valq = getFrame(sound.BUFFER, i, CLEFT);
					short int vali = getFrame(sound.BUFFER, i, CRIGHT);
					fftw.IN[i][_Q_]=((2 * (double)vali / (256 * 256)) * (fftw.HANNINGWINDOWS[i]));
					fftw.IN[i][_I_]=((2 * (double)valq / (256 * 256)) * (fftw.HANNINGWINDOWS[i]));
			}
			fftw_execute(fftw.PLAN);
			
			if(CURRENTLINE_SET){
				for (int p = 0; p < fftw.OUTLENGHT; p++)
				{
					int i = 0;
					if(p<(fftw.OUTLENGHT/2)) {i=(fftw.OUTLENGHT/2)-p;}
					else{i=(fftw.OUTLENGHT)-(p-(fftw.OUTLENGHT/2));}
					double val = sqrt(fftw.OUT[i][_Q_] * fftw.OUT[i][_Q_] + fftw.OUT[i][_I_] * fftw.OUT[i][_I_])/0.006125;
					val = val > 1.0 ? 1.0 : val;
					fftw.CURRENTLINE[p] = val;
				}
			}
			else
			{
				for (int p = 0; p < fftw.OUTLENGHT; p++)
				{
					int i = 0;
					if(p<(fftw.OUTLENGHT/2)) {i=(fftw.OUTLENGHT/2)-p;}
					else{i=(fftw.OUTLENGHT)-(p-(fftw.OUTLENGHT/2));}
					double val = sqrt(fftw.OUT[i][_Q_] * fftw.OUT[i][_Q_] + fftw.OUT[i][_I_] * fftw.OUT[i][_I_])/0.006125;
					val = val > 1.0 ? 1.0 : val;
					fftw.CURRENTLINE[p] += val;
					fftw.CURRENTLINE[p] /=2;
				}
			}
			
			CURRENTLINE_SET=0;
			
			if(!keepRunning)break;
			
			clock_gettime(CLOCK_REALTIME, &tp);
			start_stop = ((fftw.INTERVAL*1000000)/fftw.AVG)-((tp.tv_sec * 1000000 + tp.tv_nsec/1000) - start_stop);
			long_sleep(start_stop);
			
		}

		int actu_pixel_index = waterfall.TEXT_OFFSET + (actu_line_index * waterfall.width);

		if(waterfall.TIMER_MARKER_INTERVAL){
			waterfall.pixels[actu_pixel_index-1].green = 255;
		}
	
		for (int p = 0; p < fftw.OUTLENGHT; p++)
		{
			short index = fftw.CURRENTLINE[p] * 255;
			waterfall.pixels[actu_pixel_index].red   = websdrWatterfall[index][0];
			waterfall.pixels[actu_pixel_index].green = websdrWatterfall[index][1];
			waterfall.pixels[actu_pixel_index].blue  = websdrWatterfall[index][2];
			actu_pixel_index++;
		}

		if(waterfall.TIMER_MARKER_INTERVAL){
			now = time(NULL);now_tm = localtime(&now);hour = now_tm->tm_hour;min = now_tm->tm_min;sec = now_tm->tm_sec;
			if(VERBOSE)printf("the timestamp is %d:%d:%d print marker %d\n", hour,min,sec,(!(min % waterfall.TIMER_MARKER_INTERVAL) && (min != last_maker_time)));
			
			
			if(!(min % waterfall.TIMER_MARKER_INTERVAL) && (min != last_maker_time))
			{

				draw_time(now,actu_line_index);
				last_maker_time = min;
			}

		}

		
		WRITE_OUTPUT_INTERVAL_counter--;
		if(WRITE_OUTPUT_INTERVAL_counter<1){
			pthread_create(&writeBitmap_thread_id, NULL, writeBitmap, (void *)&actu_line_index); 
			WRITE_OUTPUT_INTERVAL_counter = fftw.WRITE_OUTPUT_INTERVAL;
		}
		
		actu_line_index++;
		if(actu_line_index>=waterfall.height){actu_line_index=0;}
	}

	atexit(bitmapDeinit);
	atexit(audioDeinit);
	atexit(fftwDeinit);

	exit (0);
}
