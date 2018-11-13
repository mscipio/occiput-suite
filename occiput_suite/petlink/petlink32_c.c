
// petlink - Decode and encode PETlink streams.
// Stefano Pedemonte
// Aalto University, School of Science, Helsinki
// Oct 2013, Helsinki 
// Martinos Center for Biomedical Imaging, Harvard University/MGH, Boston
// Dec. 2013, Boston

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define STATUS_SUCCESS              0
#define STATUS_IO_ERROR             1
#define STATUS_DECODE_ERROR         2 
#define STATUS_INITIALISATION_ERROR 3
#define STATUS_PARAMETER_ERROR      4
#define STATUS_UNHANDLED_ERROR      5 

/*
Return the status flags, so that they can be used by the (Python) wrapper to interpret 
the result of the function calls. 
*/
extern int status_success(int *status)
{
    *status = STATUS_SUCCESS; 
    return STATUS_SUCCESS; 
}
extern int status_io_error(int *status)
{
    *status = STATUS_IO_ERROR; 
    return STATUS_SUCCESS; 
}
extern int status_decode_error(int *status)
{
    *status = STATUS_DECODE_ERROR; 
    return STATUS_SUCCESS; 
}
extern int status_initialisation_error(int *status)
{
    *status = STATUS_INITIALISATION_ERROR; 
    return STATUS_SUCCESS; 
}
extern int status_parameter_error(int *status)
{
    *status = STATUS_PARAMETER_ERROR; 
    return STATUS_SUCCESS; 
}
extern int status_unhandled_error(int *status)
{
    *status = STATUS_UNHANDLED_ERROR; 
    return STATUS_SUCCESS; 
}



/*
Return the integer parameter. This function is meant to test 
the (Python) wrapper of the library. 
*/
extern int echo(int *in, int *out)
{
    *out = *in; 
    return STATUS_SUCCESS; 
}





#define EVENT_MASK  0b10000000
#define EVENT_VALUE 0b00000000 

#define PROMPT_MASK  0b11000000
#define PROMPT_VALUE 0b01000000 

#define TIME_MASK  0b11000000 
#define TIME_VALUE 0b10000000 

#define MOTION_MASK  0b11100000
#define MOTION_VALUE 0b11000000 

#define MONITORING_MASK  0b11110000
#define MONITORING_VALUE 0b11100000 

#define CONTROL_MASK  0b11110000
#define CONTROL_VALUE 0b11110000 


#define BIN_ADDRESS_MASK 0x3FFFFFFF



typedef struct Packet {
   char buffer[4]; 
   int is_event; 
   int is_prompt; 
   int is_time; 
   int is_motion;
   int is_monitoring;
   int is_control;
   int bin_index; 
} Packet;


int clear_packet(Packet *packet)
{
   packet->is_event=0; 
   packet->is_prompt=0; 
   packet->is_time=0; 
   packet->is_motion=0;
   packet->is_monitoring=0;
   packet->is_control=0;
   packet->bin_index=0; 
   return STATUS_SUCCESS;
}


/* 
Print n as a binary number 
*/
void printbits(int n) 
{
    unsigned int i, step;

    if (0 == n)  /* For simplicity's sake, I treat 0 as a special case*/
    {
        printf("0000");
        return;
    }

    i = 1<<(sizeof(n) * 8 - 1);

    step = -1; /* Only print the relevant digits */
    step >>= 4; /* In groups of 4 */
    while (step >= n) 
    {
        i >>= 4;
        step >>= 4;
    }

    /* At this point, i is the smallest power of two larger or equal to n */
    while (i > 0) 
    {
        if (n & i)
            printf("1");
        else
            printf("0");
        i >>= 1;
    }
    printf("\n");
}



int decode_packet(Packet* packet)
{
  //int *int_p; 
  //int_p = (int*) packet->buffer; 
  //printbits(*int_p); 

    // Detect the type of packet
    // detect bits configuration: check if the binary AND of PACKET and MASK equals VALUE: 
    if ( !((packet->buffer[3] & EVENT_MASK) - EVENT_VALUE) ) {
        packet->is_event = 1;  
        if ( !((packet->buffer[3] & PROMPT_MASK) - PROMPT_VALUE) ) {        
            packet->is_prompt = 1; 
        }        
    }
    else {
        if ( !((packet->buffer[3] & TIME_MASK) - TIME_VALUE) ) {        
            packet->is_time = 1; 
        }
        else if ( !((packet->buffer[3] & MOTION_MASK) - MOTION_VALUE) ) {        
            packet->is_motion = 1; 
        }
        else if ( !((packet->buffer[3] & MONITORING_MASK) - MONITORING_VALUE) ) {        
            packet->is_monitoring = 1; 
        }
        else if ( !((packet->buffer[3] & CONTROL_MASK) - CONTROL_VALUE) ) {        
            packet->is_control = 1; 
        }
    }
   // Read the content of the packet 
   if (packet->is_event) {
       packet->bin_index = ( *(int*)(packet->buffer) & BIN_ADDRESS_MASK); 
   }

   return STATUS_SUCCESS; 
}


int read_packet(FILE *stream, Packet* packet) 
{
    if( fread(packet->buffer,4,1,stream) != 1) 
        return STATUS_IO_ERROR; 
    return STATUS_SUCCESS; 
}


int read_and_decode_packet(FILE *stream, Packet* packet)
{
    if( read_packet(stream, packet) != STATUS_SUCCESS) 
        return STATUS_IO_ERROR; 
    return decode_packet(packet);
}


/*
Returns information about the petlink 32bit listmode binary file.
*/
extern int petlink32_info(char *filename, int *n_packets, int *n_prompts, int *n_delayed, int *n_tags, int *n_time, int *n_motion, int *n_monitoring, int *n_control) 
{
    int status = STATUS_SUCCESS; 

    // Open the petlink 32 bit listmode file
    FILE *fid; 
    fid=fopen(filename, "rb");
    if (fid == NULL) {
        fprintf(stderr,"Failed to open listmode file. \n");
        status = STATUS_IO_ERROR; 
        return status; 
    }

    // Read and decode packets 
    int i; 
    Packet packet; 
    for (i = 0; i < *n_packets; i++) {
        clear_packet(&packet);
        status = read_and_decode_packet(fid, &packet); 
        if (status!=STATUS_SUCCESS) {
            fprintf(stderr,"Problem reading and decoding the listmode data.\n"); 
            fclose(fid); 
            return status; 
        } 
        *n_prompts = *n_prompts + packet.is_prompt; 
        *n_delayed = *n_delayed + (!packet.is_prompt) * packet.is_event; 
        *n_tags    = *n_tags    + !packet.is_event;
        *n_time    = *n_time    + packet.is_time; 
        *n_motion  = *n_motion  + packet.is_motion; 
        *n_monitoring = *n_monitoring + packet.is_monitoring; 
        *n_control = *n_control + packet.is_control;
    }

    // Close file 
    fclose(fid); 
    return status; 
}



/*
Converts petlink32 listmode data to a list of bin addresses.
*/
extern int petlink32_bin_addresses(char *filename, int *n_packets, int *indexes, int *n_prompts, int *n_delayed, int *n_tags, int *n_time, int *n_motion, int *n_monitoring, int *n_control) 
{
    int status = STATUS_SUCCESS; 

    // Open the petlink 32 bit listmode file
    FILE *fid; 
    fid=fopen(filename, "rb");
    if (fid == NULL) {
        fprintf(stderr,"Failed to open listmode file. \n");
        status = STATUS_IO_ERROR; 
        return status; 
    }

    // Read and decode packets 
    int i; 
    Packet packet; 
    for (i = 0; i < *n_packets; i++) {
        clear_packet(&packet);
        status = read_and_decode_packet(fid, &packet); 
        if (status!=STATUS_SUCCESS) {
            fprintf(stderr,"Problem reading and decoding the listmode data.\n"); 
            fclose(fid); 
            return status; 
        } 
        *n_prompts = *n_prompts + packet.is_prompt; 
        *n_delayed = *n_delayed + (!packet.is_prompt) * packet.is_event; 
        *n_tags    = *n_tags    + !packet.is_event;
        *n_time    = *n_time    + packet.is_time; 
        *n_motion  = *n_motion  + packet.is_motion; 
        *n_monitoring = *n_monitoring + packet.is_monitoring; 
        *n_control = *n_control + packet.is_control;

    }

    // Close file 
    fclose(fid); 
    return status; 
}





