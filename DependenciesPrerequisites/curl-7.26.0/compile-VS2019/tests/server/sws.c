/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2012, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at http://curl.haxx.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/
#include "server_setup.h"

/* sws.c: simple (silly?) web server

   This code was originally graciously donated to the project by Juergen
   Wilke. Thanks a bunch!

 */

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_SYS_SOCKET_H
#include <sys/socket.h>
#endif
#ifdef HAVE_NETINET_IN_H
#include <netinet/in.h>
#endif
#ifdef HAVE_ARPA_INET_H
#include <arpa/inet.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef HAVE_NETINET_TCP_H
#include <netinet/tcp.h> /* for TCP_NODELAY */
#endif

#define ENABLE_CURLX_PRINTF
/* make the curlx header define all printf() functions to use the curlx_*
   versions instead */
#include "curlx.h" /* from the private lib dir */
#include "getpart.h"
#include "inet_pton.h"
#include "util.h"
#include "server_sockaddr.h"

/* include memdebug.h last */
#include "memdebug.h"

#if !defined(CURL_SWS_FORK_ENABLED) && defined(HAVE_FORK)
/*
 * The normal sws build for the plain standard curl test suite has no use for
 * fork(), but if you feel wild and crazy and want to setup some more exotic
 * tests. Define this and run...
 */
#define CURL_SWS_FORK_ENABLED
#endif

#ifdef ENABLE_IPV6
static bool use_ipv6 = FALSE;
#endif
static bool use_gopher = FALSE;
static const char *ipv_inuse = "IPv4";
static int serverlogslocked = 0;
static bool is_proxy = FALSE;

#define REQBUFSIZ 150000
#define REQBUFSIZ_TXT "149999"

static long prevtestno=-1;    /* previous test number we served */
static long prevpartno=-1;    /* previous part number we served */
static bool prevbounce=FALSE; /* instructs the server to increase the part
                                 number for a test in case the identical
                                 testno+partno request shows up again */

#define RCMD_NORMALREQ 0 /* default request, use the tests file normally */
#define RCMD_IDLE      1 /* told to sit idle */
#define RCMD_STREAM    2 /* told to stream */

struct httprequest {
  char reqbuf[REQBUFSIZ]; /* buffer area for the incoming request */
  size_t checkindex; /* where to start checking of the request */
  size_t offset;     /* size of the incoming request */
  long testno;       /* test number found in the request */
  long partno;       /* part number found in the request */
  bool open;      /* keep connection open info, as found in the request */
  bool auth_req;  /* authentication required, don't wait for body unless
                     there's an Authorization header */
  bool auth;      /* Authorization header present in the incoming request */
  size_t cl;      /* Content-Length of the incoming request */
  bool digest;    /* Authorization digest header found */
  bool ntlm;      /* Authorization ntlm header found */
  int writedelay; /* if non-zero, delay this number of seconds between
                     writes in the response */
  int pipe;       /* if non-zero, expect this many requests to do a "piped"
                     request/response */
  int skip;       /* if non-zero, the server is instructed to not read this
                     many bytes from a PUT/POST request. Ie the client sends N
                     bytes said in Content-Length, but the server only reads N
                     - skip bytes. */
  int rcmd;       /* doing a special command, see defines above */
  int prot_version;  /* HTTP version * 10 */
  bool pipelining;   /* true if request is pipelined */
  int callcount;  /* times ProcessRequest() gets called */
  unsigned short connect_port; /* the port number CONNECT used */
};

static int ProcessRequest(struct httprequest *req);
static void storerequest(char *reqbuf, size_t totalsize);

#define DEFAULT_PORT 8999

#ifndef DEFAULT_LOGFILE
#define DEFAULT_LOGFILE "log/sws.log"
#endif

const char *serverlogfile = DEFAULT_LOGFILE;

#define SWSVERSION "cURL test suite HTTP server/0.1"

#define REQUEST_DUMP  "log/server.input"
#define RESPONSE_DUMP "log/server.response"

/* when told to run as proxy, we store the logs in different files so that
   they can co-exist with the same program running as a "server" */
#define REQUEST_PROXY_DUMP  "log/proxy.input"
#define RESPONSE_PROXY_DUMP "log/proxy.response"

/* very-big-path support */
#define MAXDOCNAMELEN 140000
#define MAXDOCNAMELEN_TXT "139999"

#define REQUEST_KEYWORD_SIZE 256
#define REQUEST_KEYWORD_SIZE_TXT "255"

#define CMD_AUTH_REQUIRED "auth_required"

/* 'idle' means that it will accept the request fine but never respond
   any data. Just keep the connection alive. */
#define CMD_IDLE "idle"

/* 'stream' means to send a never-ending stream of data */
#define CMD_STREAM "stream"

#define END_OF_HEADERS "\r\n\r\n"

enum {
  DOCNUMBER_NOTHING = -7,
  DOCNUMBER_QUIT    = -6,
  DOCNUMBER_BADCONNECT = -5,
  DOCNUMBER_INTERNAL= -4,
  DOCNUMBER_CONNECT = -3,
  DOCNUMBER_WERULEZ = -2,
  DOCNUMBER_404     = -1
};

static const char *end_of_headers = END_OF_HEADERS;

/* sent as reply to a QUIT */
static const char *docquit =
"HTTP/1.1 200 Goodbye" END_OF_HEADERS;

/* sent as reply to a CONNECT */
static const char *docconnect =
"HTTP/1.1 200 Mighty fine indeed" END_OF_HEADERS;

/* sent as reply to a "bad" CONNECT */
static const char *docbadconnect =
"HTTP/1.1 501 Forbidden you fool" END_OF_HEADERS;

/* send back this on 404 file not found */
static const char *doc404 = "HTTP/1.1 404 Not Found\r\n"
    "Server: " SWSVERSION "\r\n"
    "Connection: close\r\n"
    "Content-Type: text/html"
    END_OF_HEADERS
    "<!DOCTYPE HTML PUBLIC \"-//IETF//DTD HTML 2.0//EN\">\n"
    "<HTML><HEAD>\n"
    "<TITLE>404 Not Found</TITLE>\n"
    "</HEAD><BODY>\n"
    "<H1>Not Found</H1>\n"
    "The requested URL was not found on this server.\n"
    "<P><HR><ADDRESS>" SWSVERSION "</ADDRESS>\n" "</BODY></HTML>\n";

/* do-nothing macro replacement for systems which lack siginterrupt() */

#ifndef HAVE_SIGINTERRUPT
#define siginterrupt(x,y) do {} while(0)
#endif

/* vars used to keep around previous signal handlers */

typedef RETSIGTYPE (*SIGHANDLER_T)(int);

#ifdef SIGHUP
static SIGHANDLER_T old_sighup_handler  = SIG_ERR;
#endif

#ifdef SIGPIPE
static SIGHANDLER_T old_sigpipe_handler = SIG_ERR;
#endif

#ifdef SIGALRM
static SIGHANDLER_T old_sigalrm_handler = SIG_ERR;
#endif

#ifdef SIGINT
static SIGHANDLER_T old_sigint_handler  = SIG_ERR;
#endif

#ifdef SIGTERM
static SIGHANDLER_T old_sigterm_handler = SIG_ERR;
#endif

/* var which if set indicates that the program should finish execution */

SIG_ATOMIC_T got_exit_signal = 0;

/* if next is set indicates the first signal handled in exit_signal_handler */

static volatile int exit_signal = 0;

/* signal handler that will be triggered to indicate that the program
  should finish its execution in a controlled manner as soon as possible.
  The first time this is called it will set got_exit_signal to one and
  store in exit_signal the signal that triggered its execution. */

static RETSIGTYPE exit_signal_handler(int signum)
{
  int old_errno = ERRNO;
  if(got_exit_signal == 0) {
    got_exit_signal = 1;
    exit_signal = signum;
  }
  (void)signal(signum, exit_signal_handler);
  SET_ERRNO(old_errno);
}

static void install_signal_handlers(void)
{
#ifdef SIGHUP
  /* ignore SIGHUP signal */
  if((old_sighup_handler = signal(SIGHUP, SIG_IGN)) == SIG_ERR)
    logmsg("cannot install SIGHUP handler: %s", strerror(ERRNO));
#endif
#ifdef SIGPIPE
  /* ignore SIGPIPE signal */
  if((old_sigpipe_handler = signal(SIGPIPE, SIG_IGN)) == SIG_ERR)
    logmsg("cannot install SIGPIPE handler: %s", strerror(ERRNO));
#endif
#ifdef SIGALRM
  /* ignore SIGALRM signal */
  if((old_sigalrm_handler = signal(SIGALRM, SIG_IGN)) == SIG_ERR)
    logmsg("cannot install SIGALRM handler: %s", strerror(ERRNO));
#endif
#ifdef SIGINT
  /* handle SIGINT signal with our exit_signal_handler */
  if((old_sigint_handler = signal(SIGINT, exit_signal_handler)) == SIG_ERR)
    logmsg("cannot install SIGINT handler: %s", strerror(ERRNO));
  else
    siginterrupt(SIGINT, 1);
#endif
#ifdef SIGTERM
  /* handle SIGTERM signal with our exit_signal_handler */
  if((old_sigterm_handler = signal(SIGTERM, exit_signal_handler)) == SIG_ERR)
    logmsg("cannot install SIGTERM handler: %s", strerror(ERRNO));
  else
    siginterrupt(SIGTERM, 1);
#endif
}

static void restore_signal_handlers(void)
{
#ifdef SIGHUP
  if(SIG_ERR != old_sighup_handler)
    (void)signal(SIGHUP, old_sighup_handler);
#endif
#ifdef SIGPIPE
  if(SIG_ERR != old_sigpipe_handler)
    (void)signal(SIGPIPE, old_sigpipe_handler);
#endif
#ifdef SIGALRM
  if(SIG_ERR != old_sigalrm_handler)
    (void)signal(SIGALRM, old_sigalrm_handler);
#endif
#ifdef SIGINT
  if(SIG_ERR != old_sigint_handler)
    (void)signal(SIGINT, old_sigint_handler);
#endif
#ifdef SIGTERM
  if(SIG_ERR != old_sigterm_handler)
    (void)signal(SIGTERM, old_sigterm_handler);
#endif
}

static int ProcessRequest(struct httprequest *req)
{
  char *line=&req->reqbuf[req->checkindex];
  bool chunked = FALSE;
  static char request[REQUEST_KEYWORD_SIZE];
  static char doc[MAXDOCNAMELEN];
  char logbuf[456];
  int prot_major, prot_minor;
  char *end;
  int error;
  end = strstr(line, end_of_headers);

  req->callcount++;

  logmsg("Process %d bytes request%s", req->offset,
         req->callcount > 1?" [CONTINUED]":"");

  /* try to figure out the request characteristics as soon as possible, but
     only once! */

  if(use_gopher &&
     (req->testno == DOCNUMBER_NOTHING) &&
     !strncmp("/verifiedserver", line, 15)) {
    logmsg("Are-we-friendly question received");
    req->testno = DOCNUMBER_WERULEZ;
    return 1; /* done */
  }

  else if((req->testno == DOCNUMBER_NOTHING) &&
     sscanf(line,
            "%" REQUEST_KEYWORD_SIZE_TXT"s %" MAXDOCNAMELEN_TXT "s HTTP/%d.%d",
            request,
            doc,
            &prot_major,
            &prot_minor) == 4) {
    char *ptr;

    req->prot_version = prot_major*10 + prot_minor;

    /* find the last slash */
    ptr = strrchr(doc, '/');

    /* get the number after it */
    if(ptr) {
      FILE *stream;
      char *filename;

      if((strlen(doc) + strlen(request)) < 400)
        sprintf(logbuf, "Got request: %s %s HTTP/%d.%d",
                request, doc, prot_major, prot_minor);
      else
        sprintf(logbuf, "Got a *HUGE* request HTTP/%d.%d",
                prot_major, prot_minor);
      logmsg("%s", logbuf);

      if(!strncmp("/verifiedserver", ptr, 15)) {
        logmsg("Are-we-friendly question received");
        req->testno = DOCNUMBER_WERULEZ;
        return 1; /* done */
      }

      if(!strncmp("/quit", ptr, 5)) {
        logmsg("Request-to-quit received");
        req->testno = DOCNUMBER_QUIT;
        return 1; /* done */
      }

      ptr++; /* skip the slash */

      /* skip all non-numericals following the slash */
      while(*ptr && !ISDIGIT(*ptr))
        ptr++;

      req->testno = strtol(ptr, &ptr, 10);

      if(req->testno > 10000) {
        req->partno = req->testno % 10000;
        req->testno /= 10000;
      }
      else
        req->partno = 0;

      sprintf(logbuf, "Requested test number %ld part %ld",
              req->testno, req->partno);
      logmsg("%s", logbuf);

      filename = test2file(req->testno);

      stream=fopen(filename, "rb");
      if(!stream) {
        error = ERRNO;
        logmsg("fopen() failed with error: %d %s", error, strerror(error));
        logmsg("Error opening file: %s", filename);
        logmsg("Couldn't open test file %ld", req->testno);
        req->open = FALSE; /* closes connection */
        return 1; /* done */
      }
      else {
        char *orgcmd = NULL;
        char *cmd = NULL;
        size_t cmdsize = 0;
        int num=0;

        /* get the custom server control "commands" */
        error = getpart(&orgcmd, &cmdsize, "reply", "servercmd", stream);
        fclose(stream);
        if(error) {
          logmsg("getpart() failed with error: %d", error);
          req->open = FALSE; /* closes connection */
          return 1; /* done */
        }

        cmd = orgcmd;
        while(cmd && cmdsize) {
          char *check;

          if(!strncmp(CMD_AUTH_REQUIRED, cmd, strlen(CMD_AUTH_REQUIRED))) {
            logmsg("instructed to require authorization header");
            req->auth_req = TRUE;
          }
          else if(!strncmp(CMD_IDLE, cmd, strlen(CMD_IDLE))) {
            logmsg("instructed to idle");
            req->rcmd = RCMD_IDLE;
            req->open = TRUE;
          }
          else if(!strncmp(CMD_STREAM, cmd, strlen(CMD_STREAM))) {
            logmsg("instructed to stream");
            req->rcmd = RCMD_STREAM;
          }
          else if(1 == sscanf(cmd, "pipe: %d", &num)) {
            logmsg("instructed to allow a pipe size of %d", num);
            if(num < 0)
              logmsg("negative pipe size ignored");
            else if(num > 0)
              req->pipe = num-1; /* decrease by one since we don't count the
                                    first request in this number */
          }
          else if(1 == sscanf(cmd, "skip: %d", &num)) {
            logmsg("instructed to skip this number of bytes %d", num);
            req->skip = num;
          }
          else if(1 == sscanf(cmd, "writedelay: %d", &num)) {
            logmsg("instructed to delay %d secs between packets", num);
            req->writedelay = num;
          }
          else {
            logmsg("funny instruction found: %s", cmd);
          }
          /* try to deal with CRLF or just LF */
          check = strchr(cmd, '\r');
          if(!check)
            check = strchr(cmd, '\n');

          if(check) {
            /* get to the letter following the newline */
            while((*check == '\r') || (*check == '\n'))
              check++;

            if(!*check)
              /* if we reached a zero, get out */
              break;
            cmd = check;
          }
          else
            break;
        }
        if(orgcmd)
          free(orgcmd);
      }
    }
    else {
      if(sscanf(req->reqbuf, "CONNECT %" MAXDOCNAMELEN_TXT "s HTTP/%d.%d",
                doc, &prot_major, &prot_minor) == 3) {
        char *portp = NULL;

        sprintf(logbuf, "Received a CONNECT %s HTTP/%d.%d request",
                doc, prot_major, prot_minor);
        logmsg("%s", logbuf);

        if(req->prot_version == 10)
          req->open = FALSE; /* HTTP 1.0 closes connection by default */

        if(doc[0] == '[') {
          char *p = &doc[1];
          while(*p && (ISXDIGIT(*p) || (*p == ':') || (*p == '.')))
            p++;
          if(*p != ']')
            logmsg("Invalid CONNECT IPv6 address format");
          else if (*(p+1) != ':')
            logmsg("Invalid CONNECT IPv6 port format");
          else
            portp = p+1;
        }
        else
          portp = strchr(doc, ':');

        if(portp && (*(portp+1) != '\0') && ISDIGIT(*(portp+1))) {
          unsigned long ulnum = strtoul(portp+1, NULL, 10);
          if(!ulnum || (ulnum > 65535UL))
            logmsg("Invalid CONNECT port received");
          else
            req->connect_port = curlx_ultous(ulnum);
        }

        if(!strncmp(doc, "bad", 3))
          /* if the host name starts with bad, we fake an error here */
          req->testno = DOCNUMBER_BADCONNECT;
        else if(!strncmp(doc, "test", 4))
          /* if the host name starts with test, the port number used in the
             CONNECT line will be used as test number! */
          req->testno = req->connect_port?req->connect_port:DOCNUMBER_CONNECT;
        else
          req->testno = req->connect_port?DOCNUMBER_CONNECT:DOCNUMBER_BADCONNECT;
      }
      else {
        logmsg("Did not find test number in PATH");
        req->testno = DOCNUMBER_404;
      }
    }
  }
  else if((req->offset >= 3) && (req->testno == DOCNUMBER_NOTHING)) {
    logmsg("** Unusual request. Starts with %02x %02x %02x",
           line[0], line[1], line[2]);
  }

  if(!end) {
    /* we don't have a complete request yet! */
    logmsg("request not complete yet");
    return 0; /* not complete yet */
  }
  logmsg("- request found to be complete");

  if(use_gopher) {
    /* when using gopher we cannot check the request until the entire
       thing has been received */
    char *ptr;

    /* find the last slash in the line */
    ptr = strrchr(line, '/');

    if(ptr) {
      ptr++; /* skip the slash */

      /* skip all non-numericals following the slash */
      while(*ptr && !ISDIGIT(*ptr))
        ptr++;

      req->testno = strtol(ptr, &ptr, 10);

      if(req->testno > 10000) {
        req->partno = req->testno % 10000;
        req->testno /= 10000;
      }
      else
        req->partno = 0;

      sprintf(logbuf, "Requested GOPHER test number %ld part %ld",
              req->testno, req->partno);
      logmsg("%s", logbuf);
    }
  }

  if(req->pipe)
    /* we do have a full set, advance the checkindex to after the end of the
       headers, for the pipelining case mostly */
    req->checkindex += (end - line) + strlen(end_of_headers);

  /* **** Persistence ****
   *
   * If the request is a HTTP/1.0 one, we close the connection unconditionally
   * when we're done.
   *
   * If the request is a HTTP/1.1 one, we MUST check for a "Connection:"
   * header that might say "close". If it does, we close a connection when
   * this request is processed. Otherwise, we keep the connection alive for X
   * seconds.
   */

  do {
    if(got_exit_signal)
      return 1; /* done */

    if((req->cl==0) && curlx_strnequal("Content-Length:", line, 15)) {
      /* If we don't ignore content-length, we read it and we read the whole
         request including the body before we return. If we've been told to
         ignore the content-length, we will return as soon as all headers
         have been received */
      char *endptr;
      char *ptr = line + 15;
      unsigned long clen = 0;
      while(*ptr && ISSPACE(*ptr))
        ptr++;
      endptr = ptr;
      SET_ERRNO(0);
      clen = strtoul(ptr, &endptr, 10);
      if((ptr == endptr) || !ISSPACE(*endptr) || (ERANGE == ERRNO)) {
        /* this assumes that a zero Content-Length is valid */
        logmsg("Found invalid Content-Length: (%s) in the request", ptr);
        req->open = FALSE; /* closes connection */
        return 1; /* done */
      }
      req->cl = clen - req->skip;

      logmsg("Found Content-Length: %lu in the request", clen);
      if(req->skip)
        logmsg("... but will abort after %zu bytes", req->cl);
      break;
    }
    else if(curlx_strnequal("Transfer-Encoding: chunked", line,
                            strlen("Transfer-Encoding: chunked"))) {
      /* chunked data coming in */
      chunked = TRUE;
    }

    if(chunked) {
      if(strstr(req->reqbuf, "\r\n0\r\n\r\n"))
        /* end of chunks reached */
        return 1; /* done */
      else
        return 0; /* not done */
    }

    line = strchr(line, '\n');
    if(line)
      line++;

  } while(line);

  if(!req->auth && strstr(req->reqbuf, "Authorization:")) {
    req->auth = TRUE; /* Authorization: header present! */
    if(req->auth_req)
      logmsg("Authorization header found, as required");
  }

  if(!req->digest && strstr(req->reqbuf, "Authorization: Digest")) {
    /* If the client is passing this Digest-header, we set the part number
       to 1000. Not only to spice up the complexity of this, but to make
       Digest stuff to work in the test suite. */
    req->partno += 1000;
    req->digest = TRUE; /* header found */
    logmsg("Received Digest request, sending back data %ld", req->partno);
  }
  else if(!req->ntlm &&
          strstr(req->reqbuf, "Authorization: NTLM TlRMTVNTUAAD")) {
    /* If the client is passing this type-3 NTLM header */
    req->partno += 1002;
    req->ntlm = TRUE; /* NTLM found */
    logmsg("Received NTLM type-3, sending back data %ld", req->partno);
    if(req->cl) {
      logmsg("  Expecting %zu POSTed bytes", req->cl);
    }
  }
  else if(!req->ntlm &&
          strstr(req->reqbuf, "Authorization: NTLM TlRMTVNTUAAB")) {
    /* If the client is passing this type-1 NTLM header */
    req->partno += 1001;
    req->ntlm = TRUE; /* NTLM found */
    logmsg("Received NTLM type-1, sending back data %ld", req->partno);
  }
  else if((req->partno >= 1000) &&
          strstr(req->reqbuf, "Authorization: Basic")) {
    /* If the client is passing this Basic-header and the part number is
       already >=1000, we add 1 to the part number.  This allows simple Basic
       authentication negotiation to work in the test suite. */
    req->partno += 1;
    logmsg("Received Basic request, sending back data %ld", req->partno);
  }
  if(strstr(req->reqbuf, "Connection: close"))
    req->open = FALSE; /* close connection after this request */

  if(!req->pipe &&
     req->open &&
     req->prot_version >= 11 &&
     end &&
     req->reqbuf + req->offset > end + strlen(end_of_headers) &&
     !req->cl &&
     (!strncmp(req->reqbuf, "GET", strlen("GET")) ||
      !strncmp(req->reqbuf, "HEAD", strlen("HEAD")))) {
    /* If we have a persistent connection, HTTP version >= 1.1
       and GET/HEAD request, enable pipelining. */
    req->checkindex = (end - req->reqbuf) + strlen(end_of_headers);
    req->pipelining = TRUE;
  }

  while(req->pipe) {
    if(got_exit_signal)
      return 1; /* done */
    /* scan for more header ends within this chunk */
    line = &req->reqbuf[req->checkindex];
    end = strstr(line, end_of_headers);
    if(!end)
      break;
    req->checkindex += (end - line) + strlen(end_of_headers);
    req->pipe--;
  }

  /* If authentication is required and no auth was provided, end now. This
     makes the server NOT wait for PUT/POST data and you can then make the
     test case send a rejection before any such data has been sent. Test case
     154 uses this.*/
  if(req->auth_req && !req->auth) {
    logmsg("Return early due to auth requested by none provided");
    return 1; /* done */
  }

  if(req->cl > 0) {
    if(req->cl <= req->offset - (end - req->reqbuf) - strlen(end_of_headers))
      return 1; /* done */
    else
      return 0; /* not complete yet */
  }

  return 1; /* done */
}

/* store the entire request in a file */
static void storerequest(char *reqbuf, size_t totalsize)
{
  int res;
  int error = 0;
  size_t written;
  size_t writeleft;
  FILE *dump;
  const char *dumpfile=is_proxy?REQUEST_PROXY_DUMP:REQUEST_DUMP;

  if (reqbuf == NULL)
    return;
  if (totalsize == 0)
    return;

  do {
    dump = fopen(dumpfile, "ab");
  } while ((dump == NULL) && ((error = ERRNO) == EINTR));
  if (dump == NULL) {
    logmsg("Error opening file %s error: %d %s",
           dumpfile, error, strerror(error));
    logmsg("Failed to write request input ");
    return;
  }

  writeleft = totalsize;
  do {
    written = fwrite(&reqbuf[totalsize-writeleft],
                     1, writeleft, dump);
    if(got_exit_signal)
      goto storerequest_cleanup;
    if(written > 0)
      writeleft -= written;
  } while ((writeleft > 0) && ((error = ERRNO) == EINTR));

  if(writeleft == 0)
    logmsg("Wrote request (%zu bytes) input to %s", totalsize, dumpfile);
  else if(writeleft > 0) {
    logmsg("Error writing file %s error: %d %s",
           dumpfile, error, strerror(error));
    logmsg("Wrote only (%zu bytes) of (%zu bytes) request input to %s",
           totalsize-writeleft, totalsize, dumpfile);
  }

storerequest_cleanup:

  do {
    res = fclose(dump);
  } while(res && ((error = ERRNO) == EINTR));
  if(res)
    logmsg("Error closing file %s error: %d %s",
           dumpfile, error, strerror(error));
}

/* return 0 on success, non-zero on failure */
static int get_request(curl_socket_t sock, struct httprequest *req)
{
  int error;
  int fail = 0;
  int done_processing = 0;
  char *reqbuf = req->reqbuf;
  ssize_t got = 0;

  char *pipereq = NULL;
  size_t pipereq_length = 0;

  if(req->pipelining) {
    pipereq = reqbuf + req->checkindex;
    pipereq_length = req->offset - req->checkindex;
  }

  /*** Init the httprequest structure properly for the upcoming request ***/

  req->checkindex = 0;
  req->offset = 0;
  req->testno = DOCNUMBER_NOTHING;
  req->partno = 0;
  req->open = TRUE;
  req->auth_req = FALSE;
  req->auth = FALSE;
  req->cl = 0;
  req->digest = FALSE;
  req->ntlm = FALSE;
  req->pipe = 0;
  req->skip = 0;
  req->writedelay = 0;
  req->rcmd = RCMD_NORMALREQ;
  req->prot_version = 0;
  req->pipelining = FALSE;
  req->callcount = 0;
  req->connect_port = 0;

  /*** end of httprequest init ***/

  while(!done_processing && (req->offset < REQBUFSIZ-1)) {
    if(pipereq_length && pipereq) {
      memmove(reqbuf, pipereq, pipereq_length);
      got = curlx_uztosz(pipereq_length);
      pipereq_length = 0;
    }
    else {
      if(req->skip)
        /* we are instructed to not read the entire thing, so we make sure to
           only read what we're supposed to and NOT read the enire thing the
           client wants to send! */
        got = sread(sock, reqbuf + req->offset, req->cl);
      else
        got = sread(sock, reqbuf + req->offset, REQBUFSIZ-1 - req->offset);
    }
    if(got_exit_signal)
      return 1;
    if(got == 0) {
      logmsg("Connection closed by client");
      fail = 1;
    }
    else if(got < 0) {
      error = SOCKERRNO;
      logmsg("recv() returned error: (%d) %s", error, strerror(error));
      fail = 1;
    }
    if(fail) {
      /* dump the request received so far to the external file */
      reqbuf[req->offset] = '\0';
      storerequest(reqbuf, req->offset);
      return 1;
    }

    logmsg("Read %zd bytes", got);

    req->offset += (size_t)got;
    reqbuf[req->offset] = '\0';

    done_processing = ProcessRequest(req);
    if(got_exit_signal)
      return 1;
    if(done_processing && req->pipe) {
      logmsg("Waiting for another piped request");
      done_processing = 0;
      req->pipe--;
    }
  }

  if((req->offset == REQBUFSIZ-1) && (got > 0)) {
    logmsg("Request would overflow buffer, closing connection");
    /* dump request received so far to external file anyway */
    reqbuf[REQBUFSIZ-1] = '\0';
    fail = 1;
  }
  else if(req->offset > REQBUFSIZ-1) {
    logmsg("Request buffer overflow, closing connection");
    /* dump request received so far to external file anyway */
    reqbuf[REQBUFSIZ-1] = '\0';
    fail = 1;
  }
  else
    reqbuf[req->offset] = '\0';

  /* dump the request to an external file */
  storerequest(reqbuf, req->pipelining ? req->checkindex : req->offset);
  if(got_exit_signal)
    return 1;

  return fail; /* return 0 on success */
}

/* returns -1 on failure */
static int send_doc(curl_socket_t sock, struct httprequest *req)
{
  ssize_t written;
  size_t count;
  const char *buffer;
  char *ptr=NULL;
  FILE *stream;
  char *cmd=NULL;
  size_t cmdsize=0;
  FILE *dump;
  bool persistant = TRUE;
  bool sendfailure = FALSE;
  size_t responsesize;
  int error = 0;
  int res;
  const char *responsedump = is_proxy?RESPONSE_PROXY_DUMP:RESPONSE_DUMP;
  static char weare[256];

  char partbuf[80]="data";

  logmsg("Send response test%ld section <data%ld>", req->testno, req->partno);

  switch(req->rcmd) {
  default:
  case RCMD_NORMALREQ:
    break; /* continue with business as usual */
  case RCMD_STREAM:
#define STREAMTHIS "a string to stream 01234567890\n"
    count = strlen(STREAMTHIS);
    for (;;) {
      written = swrite(sock, STREAMTHIS, count);
      if(got_exit_signal)
        return -1;
      if(written != (ssize_t)count) {
        logmsg("Stopped streaming");
        break;
      }
    }
    return -1;
  case RCMD_IDLE:
    /* Do nothing. Sit idle. Pretend it rains. */
    return 0;
  }

  req->open = FALSE;

  if(req->testno < 0) {
    size_t msglen;
    char msgbuf[64];

    switch(req->testno) {
    case DOCNUMBER_QUIT:
      logmsg("Replying to QUIT");
      buffer = docquit;
      break;
    case DOCNUMBER_WERULEZ:
      /* we got a "friends?" question, reply back that we sure are */
      logmsg("Identifying ourselves as friends");
      sprintf(msgbuf, "WE ROOLZ: %ld\r\n", (long)getpid());
      msglen = strlen(msgbuf);
      if(use_gopher)
        sprintf(weare, "%s", msgbuf);
      else
        sprintf(weare, "HTTP/1.1 200 OK\r\nContent-Length: %zu\r\n\r\n%s",
                msglen, msgbuf);
      buffer = weare;
      break;
    case DOCNUMBER_INTERNAL:
      logmsg("Bailing out due to internal error");
      return -1;
    case DOCNUMBER_CONNECT:
      logmsg("Replying to CONNECT");
      buffer = docconnect;
      break;
    case DOCNUMBER_BADCONNECT:
      logmsg("Replying to a bad CONNECT");
      buffer = docbadconnect;
      break;
    case DOCNUMBER_404:
    default:
      logmsg("Replying to with a 404");
      buffer = doc404;
      break;
    }

    count = strlen(buffer);
  }
  else {
    char *filename = test2file(req->testno);

    if(0 != req->partno)
      sprintf(partbuf, "data%ld", req->partno);

    stream=fopen(filename, "rb");
    if(!stream) {
      error = ERRNO;
      logmsg("fopen() failed with error: %d %s", error, strerror(error));
      logmsg("Error opening file: %s", filename);
      logmsg("Couldn't open test file");
      return 0;
    }
    else {
      error = getpart(&ptr, &count, "reply", partbuf, stream);
      fclose(stream);
      if(error) {
        logmsg("getpart() failed with error: %d", error);
        return 0;
      }
      buffer = ptr;
    }

    if(got_exit_signal) {
      if(ptr)
        free(ptr);
      return -1;
    }

    /* re-open the same file again */
    stream=fopen(filename, "rb");
    if(!stream) {
      error = ERRNO;
      logmsg("fopen() failed with error: %d %s", error, strerror(error));
      logmsg("Error opening file: %s", filename);
      logmsg("Couldn't open test file");
      if(ptr)
        free(ptr);
      return 0;
    }
    else {
      /* get the custom server control "commands" */
      error = getpart(&cmd, &cmdsize, "reply", "postcmd", stream);
      fclose(stream);
      if(error) {
        logmsg("getpart() failed with error: %d", error);
        if(ptr)
          free(ptr);
        return 0;
      }
    }
  }

  if(got_exit_signal) {
    if(ptr)
      free(ptr);
    if(cmd)
      free(cmd);
    return -1;
  }

  /* If the word 'swsclose' is present anywhere in the reply chunk, the
     connection will be closed after the data has been sent to the requesting
     client... */
  if(strstr(buffer, "swsclose") || !count) {
    persistant = FALSE;
    logmsg("connection close instruction \"swsclose\" found in response");
  }
  if(strstr(buffer, "swsbounce")) {
    prevbounce = TRUE;
    logmsg("enable \"swsbounce\" in the next request");
  }
  else
    prevbounce = FALSE;

  dump = fopen(responsedump, "ab");
  if(!dump) {
    error = ERRNO;
    logmsg("fopen() failed with error: %d %s", error, strerror(error));
    logmsg("Error opening file: %s", responsedump);
    if(ptr)
      free(ptr);
    if(cmd)
      free(cmd);
    return -1;
  }

  responsesize = count;
  do {
    /* Ok, we send no more than 200 bytes at a time, just to make sure that
       larger chunks are split up so that the client will need to do multiple
       recv() calls to get it and thus we exercise that code better */
    size_t num = count;
    if(num > 200)
      num = 200;
    written = swrite(sock, buffer, num);
    if (written < 0) {
      sendfailure = TRUE;
      break;
    }
    else {
      logmsg("Sent off %zd bytes", written);
    }
    /* write to file as well */
    fwrite(buffer, 1, (size_t)written, dump);

    count -= written;
    buffer += written;

    if(req->writedelay) {
      int quarters = req->writedelay * 4;
      logmsg("Pausing %d seconds", req->writedelay);
      while((quarters > 0) && !got_exit_signal) {
        quarters--;
        wait_ms(250);
      }
    }
  } while((count > 0) && !got_exit_signal);

  do {
    res = fclose(dump);
  } while(res && ((error = ERRNO) == EINTR));
  if(res)
    logmsg("Error closing file %s error: %d %s",
           responsedump, error, strerror(error));

  if(got_exit_signal) {
    if(ptr)
      free(ptr);
    if(cmd)
      free(cmd);
    return -1;
  }

  if(sendfailure) {
    logmsg("Sending response failed. Only (%zu bytes) of (%zu bytes) were sent",
           responsesize-count, responsesize);
    if(ptr)
      free(ptr);
    if(cmd)
      free(cmd);
    return -1;
  }

  logmsg("Response sent (%zu bytes) and written to %s",
         responsesize, responsedump);

  if(ptr)
    free(ptr);

  if(cmdsize > 0 ) {
    char command[32];
    int quarters;
    int num;
    ptr=cmd;
    do {
      if(2 == sscanf(ptr, "%31s %d", command, &num)) {
        if(!strcmp("wait", command)) {
          logmsg("Told to sleep for %d seconds", num);
          quarters = num * 4;
          while((quarters > 0) && !got_exit_signal) {
            quarters--;
            res = wait_ms(250);
            if(res) {
              /* should not happen */
              error = SOCKERRNO;
              logmsg("wait_ms() failed with error: (%d) %s",
                     error, strerror(error));
              break;
            }
          }
          if(!quarters)
            logmsg("Continuing after sleeping %d seconds", num);
        }
        else
          logmsg("Unknown command in reply command section");
      }
      ptr = strchr(ptr, '\n');
      if(ptr)
        ptr++;
      else
        ptr = NULL;
    } while(ptr && *ptr);
  }
  if(cmd)
    free(cmd);

  req->open = use_gopher?FALSE:persistant;

  prevtestno = req->testno;
  prevpartno = req->partno;

  return 0;
}

static curl_socket_t connect_to(const char *ipaddr, unsigned short port)
{
  srvr_sockaddr_union_t serveraddr;
  curl_socket_t serverfd;
  int error;
  int rc;
  const char *op_br = "";
  const char *cl_br = "";
#ifdef TCP_NODELAY
  curl_socklen_t flag;
#endif

#ifdef ENABLE_IPV6
  if(use_ipv6) {
    op_br = "[";
    cl_br = "]";
  }
#endif

  if(!ipaddr)
    return CURL_SOCKET_BAD;

  logmsg("about to connect to %s%s%s:%hu",
         op_br, ipaddr, cl_br, port);

#ifdef ENABLE_IPV6
  if(!use_ipv6)
#endif
    serverfd = socket(AF_INET, SOCK_STREAM, 0);
#ifdef ENABLE_IPV6
  else
    serverfd = socket(AF_INET6, SOCK_STREAM, 0);
#endif
  if(CURL_SOCKET_BAD == serverfd) {
    error = SOCKERRNO;
    logmsg("Error creating socket for server conection: (%d) %s",
           error, strerror(error));
    return CURL_SOCKET_BAD;
  }

#ifdef TCP_NODELAY
  /* Disable the Nagle algorithm */
  flag = 1;
  if(0 != setsockopt(serverfd, IPPROTO_TCP, TCP_NODELAY,
                     (void *)&flag, sizeof(flag)))
    logmsg("====> TCP_NODELAY for server conection failed");
  else
    logmsg("TCP_NODELAY set for server conection");
#endif

#ifdef ENABLE_IPV6
  if(!use_ipv6) {
#endif
    memset(&serveraddr.sa4, 0, sizeof(serveraddr.sa4));
    serveraddr.sa4.sin_family = AF_INET;
    serveraddr.sa4.sin_port = htons(port);
    if(Curl_inet_pton(AF_INET, ipaddr, &serveraddr.sa4.sin_addr) < 1) {
      logmsg("Error inet_pton failed AF_INET conversion of '%s'", ipaddr);
      sclose(serverfd);
      return CURL_SOCKET_BAD;
    }

    rc = connect(serverfd, &serveraddr.sa, sizeof(serveraddr.sa4));
#ifdef ENABLE_IPV6
  }
  else {
    memset(&serveraddr.sa6, 0, sizeof(serveraddr.sa6));
    serveraddr.sa6.sin6_family = AF_INET6;
    serveraddr.sa6.sin6_port = htons(port);
    if(Curl_inet_pton(AF_INET6, ipaddr, &serveraddr.sa6.sin6_addr) < 1) {
      logmsg("Error inet_pton failed AF_INET6 conversion of '%s'", ipaddr);
      sclose(serverfd);
      return CURL_SOCKET_BAD;
    }

    rc = connect(serverfd, &serveraddr.sa, sizeof(serveraddr.sa6));
  }
#endif /* ENABLE_IPV6 */

  if(got_exit_signal) {
    sclose(serverfd);
    return CURL_SOCKET_BAD;
  }

  if(rc) {
    error = SOCKERRNO;
    logmsg("Error connecting to server port %hu: (%d) %s",
           port, error, strerror(error));
    sclose(serverfd);
    return CURL_SOCKET_BAD;
  }

  logmsg("connected fine to %s%s%s:%hu, now tunnel",
         op_br, ipaddr, cl_br, port);

  return serverfd;
}

/*
 * A CONNECT has been received, a CONNECT response has been sent.
 *
 * This function needs to connect to the server, and then pass data between
 * the client and the server back and forth until the connection is closed by
 * either end.
 *
 * When doing FTP through a CONNECT proxy, we expect that the data connection
 * will be setup while the first connect is still being kept up. Therefor we
 * must accept a new connection and deal with it appropriately.
 */

#define data_or_ctrl(x) ((x)?"DATA":"CTRL")

#define CTRL  0
#define DATA  1

static void http_connect(curl_socket_t *infdp,
                         curl_socket_t rootfd,
                         struct httprequest *req,
                         const char *ipaddr)
{
  curl_socket_t serverfd[2] = {CURL_SOCKET_BAD, CURL_SOCKET_BAD};
  curl_socket_t clientfd[2] = {CURL_SOCKET_BAD, CURL_SOCKET_BAD};
  ssize_t toc[2] = {0, 0}; /* number of bytes to client */
  ssize_t tos[2] = {0, 0}; /* number of bytes to server */
  char readclient[2][256];
  char readserver[2][256];
  bool poll_client_rd[2] = { TRUE, TRUE };
  bool poll_server_rd[2] = { TRUE, TRUE };
  bool poll_client_wr[2] = { TRUE, TRUE };
  bool poll_server_wr[2] = { TRUE, TRUE };
#ifdef TCP_NODELAY
  curl_socklen_t flag;
#endif
  bool primary = FALSE;
  bool secondary = FALSE;
  int max_tunnel_idx; /* CTRL or DATA */
  int loop;
  int i;

  /* primary tunnel client endpoint already connected */
  clientfd[CTRL] = *infdp;

  /* Sleep here to make sure the client reads CONNECT response's
     'end of headers' separate from the server data that follows.
     This is done to prevent triggering libcurl known bug #39. */
  for(loop = 2; (loop > 0) && !got_exit_signal; loop--)
    wait_ms(250);
  if(got_exit_signal)
    goto http_connect_cleanup;

  serverfd[CTRL] = connect_to(ipaddr, req->connect_port);
  if(serverfd[CTRL] == CURL_SOCKET_BAD)
    goto http_connect_cleanup;

  /* Primary tunnel socket endpoints are now connected. Tunnel data back and
     forth over the primary tunnel until client or server breaks the primary
     tunnel, simultaneously allowing establishment, operation and teardown of
     a secondary tunnel that may be used for passive FTP data connection. */

  max_tunnel_idx = CTRL;
  primary = TRUE;

  while(!got_exit_signal) {

    fd_set input;
    fd_set output;
    struct timeval timeout = {0, 250000L}; /* 250 ms */
    ssize_t rc;
    curl_socket_t maxfd = (curl_socket_t)-1;

    FD_ZERO(&input);
    FD_ZERO(&output);

    if((clientfd[DATA] == CURL_SOCKET_BAD) &&
       (serverfd[DATA] == CURL_SOCKET_BAD) &&
       poll_client_rd[CTRL] && poll_client_wr[CTRL] &&
       poll_server_rd[CTRL] && poll_server_wr[CTRL]) {
      /* listener socket is monitored to allow client to establish
         secondary tunnel only when this tunnel is not established
         and primary one is fully operational */
      FD_SET(rootfd, &input);
      maxfd = rootfd;
    }

    /* set tunnel sockets to wait for */
    for(i = 0; i <= max_tunnel_idx; i++) {
      /* client side socket monitoring */
      if(clientfd[i] != CURL_SOCKET_BAD) {
        if(poll_client_rd[i]) {
          /* unless told not to do so, monitor readability */
          FD_SET(clientfd[i], &input);
          if(clientfd[i] > maxfd)
            maxfd = clientfd[i];
        }
        if(poll_client_wr[i] && toc[i]) {
          /* unless told not to do so, monitor writeability
             if there is data ready to be sent to client */
          FD_SET(clientfd[i], &output);
          if(clientfd[i] > maxfd)
            maxfd = clientfd[i];
        }
      }
      /* server side socket monitoring */
      if(serverfd[i] != CURL_SOCKET_BAD) {
        if(poll_server_rd[i]) {
          /* unless told not to do so, monitor readability */
          FD_SET(serverfd[i], &input);
          if(serverfd[i] > maxfd)
            maxfd = serverfd[i];
        }
        if(poll_server_wr[i] && tos[i]) {
          /* unless told not to do so, monitor writeability
             if there is data ready to be sent to server */
          FD_SET(serverfd[i], &output);
          if(serverfd[i] > maxfd)
            maxfd = serverfd[i];
        }
      }
    }
    if(got_exit_signal)
      break;

    rc = select((int)maxfd + 1, &input, &output, NULL, &timeout);

    if(rc > 0) {
      /* socket action */
      bool tcp_fin_wr;

      if(got_exit_signal)
        break;

      tcp_fin_wr = FALSE;

      /* ---------------------------------------------------------- */

      /* passive mode FTP may establish a secondary tunnel */
      if((clientfd[DATA] == CURL_SOCKET_BAD) &&
         (serverfd[DATA] == CURL_SOCKET_BAD) && FD_ISSET(rootfd, &input)) {
        /* a new connection on listener socket (most likely from client) */
        curl_socket_t datafd = accept(rootfd, NULL, NULL);
        if(datafd != CURL_SOCKET_BAD) {
          struct httprequest req2;
          int err;
          logmsg("====> Client connect DATA");
#ifdef TCP_NODELAY
          /* Disable the Nagle algorithm */
          flag = 1;
          if(0 != setsockopt(datafd, IPPROTO_TCP, TCP_NODELAY,
                             (void *)&flag, sizeof(flag)))
            logmsg("====> TCP_NODELAY for client DATA conection failed");
          else
            logmsg("TCP_NODELAY set for client DATA conection");
#endif
          req2.pipelining = FALSE;
          err = get_request(datafd, &req2);
          if(!err) {
            err = send_doc(datafd, &req2);
            if(!err && (req2.testno == DOCNUMBER_CONNECT)) {
              /* sleep to prevent triggering libcurl known bug #39. */
              for(loop = 2; (loop > 0) && !got_exit_signal; loop--)
                wait_ms(250);
              if(!got_exit_signal) {
                /* connect to the server */
                serverfd[DATA] = connect_to(ipaddr, req2.connect_port);
                if(serverfd[DATA] != CURL_SOCKET_BAD) {
                  /* secondary tunnel established, now we have two connections */
                  poll_client_rd[DATA] = TRUE;
                  poll_client_wr[DATA] = TRUE;
                  poll_server_rd[DATA] = TRUE;
                  poll_server_wr[DATA] = TRUE;
                  max_tunnel_idx = DATA;
                  secondary = TRUE;
                  toc[DATA] = 0;
                  tos[DATA] = 0;
                  clientfd[DATA] = datafd;
                  datafd = CURL_SOCKET_BAD;
                }
              }
            }
          }
          if(datafd != CURL_SOCKET_BAD) {
            /* secondary tunnel not established */
            shutdown(datafd, SHUT_RDWR);
            sclose(datafd);
          }
        }
        if(got_exit_signal)
          break;
      }

      /* ---------------------------------------------------------- */

      /* react to tunnel endpoint readable/writeable notifications */
      for(i = 0; i <= max_tunnel_idx; i++) {
        size_t len;
        if(clientfd[i] != CURL_SOCKET_BAD) {
          len = sizeof(readclient[i]) - tos[i];
          if(len && FD_ISSET(clientfd[i], &input)) {
            /* read from client */
            rc = sread(clientfd[i], &readclient[i][tos[i]], len);
            if(rc <= 0) {
              logmsg("[%s] got %zd, STOP READING client", data_or_ctrl(i), rc);
              shutdown(clientfd[i], SHUT_RD);
              poll_client_rd[i] = FALSE;
            }
            else {
              logmsg("[%s] READ %zd bytes from client", data_or_ctrl(i), rc);
              logmsg("[%s] READ \"%s\"", data_or_ctrl(i),
                     data_to_hex(&readclient[i][tos[i]], rc));
              tos[i] += rc;
            }
          }
        }
        if(serverfd[i] != CURL_SOCKET_BAD) {
          len = sizeof(readserver[i])-toc[i];
          if(len && FD_ISSET(serverfd[i], &input)) {
            /* read from server */
            rc = sread(serverfd[i], &readserver[i][toc[i]], len);
            if(rc <= 0) {
              logmsg("[%s] got %zd, STOP READING server", data_or_ctrl(i), rc);
              shutdown(serverfd[i], SHUT_RD);
              poll_server_rd[i] = FALSE;
            }
            else {
              logmsg("[%s] READ %zd bytes from server", data_or_ctrl(i), rc);
              logmsg("[%s] READ \"%s\"", data_or_ctrl(i),
                     data_to_hex(&readserver[i][toc[i]], rc));
              toc[i] += rc;
            }
          }
        }
        if(clientfd[i] != CURL_SOCKET_BAD) {
          if(toc[i] && FD_ISSET(clientfd[i], &output)) {
            /* write to client */
            rc = swrite(clientfd[i], readserver[i], toc[i]);
            if(rc <= 0) {
              logmsg("[%s] got %zd, STOP WRITING client", data_or_ctrl(i), rc);
              shutdown(clientfd[i], SHUT_WR);
              poll_client_wr[i] = FALSE;
              tcp_fin_wr = TRUE;
            }
            else {
              logmsg("[%s] SENT %zd bytes to client", data_or_ctrl(i), rc);
              logmsg("[%s] SENT \"%s\"", data_or_ctrl(i),
                     data_to_hex(readserver[i], rc));
              if(toc[i] - rc)
                memmove(&readserver[i][0], &readserver[i][rc], toc[i]-rc);
              toc[i] -= rc;
            }
          }
        }
        if(serverfd[i] != CURL_SOCKET_BAD) {
          if(tos[i] && FD_ISSET(serverfd[i], &output)) {
            /* write to server */
            rc = swrite(serverfd[i], readclient[i], tos[i]);
            if(rc <= 0) {
              logmsg("[%s] got %zd, STOP WRITING server", data_or_ctrl(i), rc);
              shutdown(serverfd[i], SHUT_WR);
              poll_server_wr[i] = FALSE;
              tcp_fin_wr = TRUE;
            }
            else {
              logmsg("[%s] SENT %zd bytes to server", data_or_ctrl(i), rc);
              logmsg("[%s] SENT \"%s\"", data_or_ctrl(i),
                     data_to_hex(readclient[i], rc));
              if(tos[i] - rc)
                memmove(&readclient[i][0], &readclient[i][rc], tos[i]-rc);
              tos[i] -= rc;
            }
          }
        }
      }
      if(got_exit_signal)
        break;

      /* ---------------------------------------------------------- */

      /* endpoint read/write disabling, endpoint closing and tunnel teardown */
      for(i = 0; i <= max_tunnel_idx; i++) {
        for(loop = 2; loop > 0; loop--) {
          /* loop twice to satisfy condition interdependencies without
             having to await select timeout or another socket event */
          if(clientfd[i] != CURL_SOCKET_BAD) {
            if(poll_client_rd[i] && !poll_server_wr[i]) {
              logmsg("[%s] DISABLED READING client", data_or_ctrl(i));
              shutdown(clientfd[i], SHUT_RD);
              poll_client_rd[i] = FALSE;
            }
            if(poll_client_wr[i] && !poll_server_rd[i] && !toc[i]) {
              logmsg("[%s] DISABLED WRITING client", data_or_ctrl(i));
              shutdown(clientfd[i], SHUT_WR);
              poll_client_wr[i] = FALSE;
              tcp_fin_wr = TRUE;
            }
          }
          if(serverfd[i] != CURL_SOCKET_BAD) {
            if(poll_server_rd[i] && !poll_client_wr[i]) {
              logmsg("[%s] DISABLED READING server", data_or_ctrl(i));
              shutdown(serverfd[i], SHUT_RD);
              poll_server_rd[i] = FALSE;
            }
            if(poll_server_wr[i] && !poll_client_rd[i] && !tos[i]) {
              logmsg("[%s] DISABLED WRITING server", data_or_ctrl(i));
              shutdown(serverfd[i], SHUT_WR);
              poll_server_wr[i] = FALSE;
              tcp_fin_wr = TRUE;
            }
          }
        }
      }

      if(tcp_fin_wr)
        /* allow kernel to place FIN bit packet on the wire */
        wait_ms(250);

      /* socket clearing */
      for(i = 0; i <= max_tunnel_idx; i++) {
        for(loop = 2; loop > 0; loop--) {
          if(clientfd[i] != CURL_SOCKET_BAD) {
            if(!poll_client_wr[i] && !poll_client_rd[i]) {
              logmsg("[%s] CLOSING client socket", data_or_ctrl(i));
              sclose(clientfd[i]);
              clientfd[i] = CURL_SOCKET_BAD;
              if(serverfd[i] == CURL_SOCKET_BAD) {
                logmsg("[%s] ENDING", data_or_ctrl(i));
                if(i == DATA)
                  secondary = FALSE;
                else
                  primary = FALSE;
              }
            }
          }
          if(serverfd[i] != CURL_SOCKET_BAD) {
            if(!poll_server_wr[i] && !poll_server_rd[i]) {
              logmsg("[%s] CLOSING server socket", data_or_ctrl(i));
              sclose(serverfd[i]);
              serverfd[i] = CURL_SOCKET_BAD;
              if(clientfd[i] == CURL_SOCKET_BAD) {
                logmsg("[%s] ENDING", data_or_ctrl(i));
                if(i == DATA)
                  secondary = FALSE;
                else
                  primary = FALSE;
              }
            }
          }
        }
      }

      /* ---------------------------------------------------------- */

      max_tunnel_idx = secondary ? DATA : CTRL;

      if(!primary)
        /* exit loop upon primary tunnel teardown */
        break;

    } /* (rc > 0) */

  }

http_connect_cleanup:

  for(i = DATA; i >= CTRL; i--) {
    if(serverfd[i] != CURL_SOCKET_BAD) {
      logmsg("[%s] CLOSING server socket (cleanup)", data_or_ctrl(i));
      shutdown(serverfd[i], SHUT_RDWR);
      sclose(serverfd[i]);
    }
    if(clientfd[i] != CURL_SOCKET_BAD) {
      logmsg("[%s] CLOSING client socket (cleanup)", data_or_ctrl(i));
      shutdown(clientfd[i], SHUT_RDWR);
      sclose(clientfd[i]);
    }
    if((serverfd[i] != CURL_SOCKET_BAD) ||
       (clientfd[i] != CURL_SOCKET_BAD)) {
      logmsg("[%s] ABORTING", data_or_ctrl(i));
    }
  }

  *infdp = CURL_SOCKET_BAD;
}

int main(int argc, char *argv[])
{
  srvr_sockaddr_union_t me;
  curl_socket_t sock = CURL_SOCKET_BAD;
  curl_socket_t msgsock = CURL_SOCKET_BAD;
  int wrotepidfile = 0;
  int flag;
  unsigned short port = DEFAULT_PORT;
  char *pidname= (char *)".http.pid";
  struct httprequest req;
  int rc;
  int error;
  int arg=1;
  long pid;
  const char *hostport = "127.0.0.1";
#ifdef CURL_SWS_FORK_ENABLED
  bool use_fork = FALSE;
#endif

  while(argc>arg) {
    if(!strcmp("--version", argv[arg])) {
      printf("sws IPv4%s"
#ifdef CURL_SWS_FORK_ENABLED
             " FORK"
#endif
             "\n"
             ,
#ifdef ENABLE_IPV6
             "/IPv6"
#else
             ""
#endif
             );
      return 0;
    }
    else if(!strcmp("--pidfile", argv[arg])) {
      arg++;
      if(argc>arg)
        pidname = argv[arg++];
    }
    else if(!strcmp("--logfile", argv[arg])) {
      arg++;
      if(argc>arg)
        serverlogfile = argv[arg++];
    }
    else if(!strcmp("--gopher", argv[arg])) {
      arg++;
      use_gopher = TRUE;
      end_of_headers = "\r\n"; /* gopher style is much simpler */
    }
    else if(!strcmp("--ipv4", argv[arg])) {
#ifdef ENABLE_IPV6
      ipv_inuse = "IPv4";
      use_ipv6 = FALSE;
#endif
      arg++;
    }
    else if(!strcmp("--ipv6", argv[arg])) {
#ifdef ENABLE_IPV6
      ipv_inuse = "IPv6";
      use_ipv6 = TRUE;
#endif
      arg++;
    }
#ifdef CURL_SWS_FORK_ENABLED
    else if(!strcmp("--fork", argv[arg])) {
      use_fork=TRUE;
      arg++;
    }
#endif
    else if(!strcmp("--port", argv[arg])) {
      arg++;
      if(argc>arg) {
        char *endptr;
        unsigned long ulnum = strtoul(argv[arg], &endptr, 10);
        if((endptr != argv[arg] + strlen(argv[arg])) ||
           (ulnum < 1025UL) || (ulnum > 65535UL)) {
          fprintf(stderr, "sws: invalid --port argument (%s)\n",
                  argv[arg]);
          return 0;
        }
        port = curlx_ultous(ulnum);
        arg++;
      }
    }
    else if(!strcmp("--srcdir", argv[arg])) {
      arg++;
      if(argc>arg) {
        path = argv[arg];
        arg++;
      }
    }
    else if(!strcmp("--connect", argv[arg])) {
      /* store the connect host, but also use this as a hint that we
         run as a proxy and do a few different internal choices */
      arg++;
      if(argc>arg) {
        hostport = argv[arg];
        arg++;
        is_proxy = TRUE;
        logmsg("Run as proxy, CONNECT to %s", hostport);
      }
    }
    else {
      puts("Usage: sws [option]\n"
           " --version\n"
           " --logfile [file]\n"
           " --pidfile [file]\n"
           " --ipv4\n"
           " --ipv6\n"
           " --port [port]\n"
           " --srcdir [path]\n"
           " --connect [ip4-addr]\n"
           " --gopher\n"
           " --fork");
      return 0;
    }
  }

#ifdef WIN32
  win32_init();
  atexit(win32_cleanup);
#endif

  install_signal_handlers();

  pid = (long)getpid();

#ifdef ENABLE_IPV6
  if(!use_ipv6)
#endif
    sock = socket(AF_INET, SOCK_STREAM, 0);
#ifdef ENABLE_IPV6
  else
    sock = socket(AF_INET6, SOCK_STREAM, 0);
#endif

  if(CURL_SOCKET_BAD == sock) {
    error = SOCKERRNO;
    logmsg("Error creating socket: (%d) %s",
           error, strerror(error));
    goto sws_cleanup;
  }

  flag = 1;
  if(0 != setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                     (void *)&flag, sizeof(flag))) {
    error = SOCKERRNO;
    logmsg("setsockopt(SO_REUSEADDR) failed with error: (%d) %s",
           error, strerror(error));
    goto sws_cleanup;
  }

#ifdef ENABLE_IPV6
  if(!use_ipv6) {
#endif
    memset(&me.sa4, 0, sizeof(me.sa4));
    me.sa4.sin_family = AF_INET;
    me.sa4.sin_addr.s_addr = INADDR_ANY;
    me.sa4.sin_port = htons(port);
    rc = bind(sock, &me.sa, sizeof(me.sa4));
#ifdef ENABLE_IPV6
  }
  else {
    memset(&me.sa6, 0, sizeof(me.sa6));
    me.sa6.sin6_family = AF_INET6;
    me.sa6.sin6_addr = in6addr_any;
    me.sa6.sin6_port = htons(port);
    rc = bind(sock, &me.sa, sizeof(me.sa6));
  }
#endif /* ENABLE_IPV6 */
  if(0 != rc) {
    error = SOCKERRNO;
    logmsg("Error binding socket on port %hu: (%d) %s",
           port, error, strerror(error));
    goto sws_cleanup;
  }

  logmsg("Running %s %s version on port %d",
         use_gopher?"GOPHER":"HTTP", ipv_inuse, (int)port);

  /* start accepting connections */
  rc = listen(sock, 5);
  if(0 != rc) {
    error = SOCKERRNO;
    logmsg("listen() failed with error: (%d) %s",
           error, strerror(error));
    goto sws_cleanup;
  }

  /*
  ** As soon as this server writes its pid file the test harness will
  ** attempt to connect to this server and initiate its verification.
  */

  wrotepidfile = write_pidfile(pidname);
  if(!wrotepidfile)
    goto sws_cleanup;

  for (;;) {
    msgsock = accept(sock, NULL, NULL);

    if(got_exit_signal)
      break;
    if (CURL_SOCKET_BAD == msgsock) {
      error = SOCKERRNO;
      logmsg("MAJOR ERROR: accept() failed with error: (%d) %s",
             error, strerror(error));
      break;
    }

    /*
    ** As soon as this server acepts a connection from the test harness it
    ** must set the server logs advisor read lock to indicate that server
    ** logs should not be read until this lock is removed by this server.
    */

    set_advisor_read_lock(SERVERLOGS_LOCK);
    serverlogslocked = 1;

#ifdef CURL_SWS_FORK_ENABLED
    if(use_fork) {
      /* The fork enabled version just forks off the child and don't care
         about it anymore, so don't assume otherwise. Beware and don't do
         this at home. */
      rc = fork();
      if(-1 == rc) {
        printf("MAJOR ERROR: fork() failed!\n");
        break;
      }
    }
    else
      /* not a fork, just set rc so the following proceeds nicely */
      rc = 0;
    /* 0 is returned to the child */
    if(0 == rc) {
#endif
    logmsg("====> Client connect");

#ifdef TCP_NODELAY
    /*
     * Disable the Nagle algorithm to make it easier to send out a large
     * response in many small segments to torture the clients more.
     */
    flag = 1;
    if(0 != setsockopt(msgsock, IPPROTO_TCP, TCP_NODELAY,
                       (void *)&flag, sizeof(flag)))
      logmsg("====> TCP_NODELAY failed");
    else
      logmsg("TCP_NODELAY set");
#endif

    /* initialization of httprequest struct is done in get_request(), but due
       to pipelining treatment the pipelining struct field must be initialized
       previously to FALSE every time a new connection arrives. */

    req.pipelining = FALSE;

    do {
      if(got_exit_signal)
        break;

      if(get_request(msgsock, &req))
        /* non-zero means error, break out of loop */
        break;

      if(prevbounce) {
        /* bounce treatment requested */
        if((req.testno == prevtestno) &&
           (req.partno == prevpartno)) {
          req.partno++;
          logmsg("BOUNCE part number to %ld", req.partno);
        }
        else {
          prevbounce = FALSE;
          prevtestno = -1;
          prevpartno = -1;
        }
      }

      send_doc(msgsock, &req);
      if(got_exit_signal)
        break;

      if(DOCNUMBER_CONNECT == req.testno) {
        /* a CONNECT request, setup and talk the tunnel */
        if(!is_proxy) {
          logmsg("received CONNECT but isn't running as proxy! EXIT");
        }
        else
          http_connect(&msgsock, sock, &req, hostport);
        break;
      }

      if((req.testno < 0) && (req.testno != DOCNUMBER_CONNECT)) {
        logmsg("special request received, no persistency");
        break;
      }
      if(!req.open) {
        logmsg("instructed to close connection after server-reply");
        break;
      }

      if(req.open) {
        logmsg("=> persistant connection request ended, awaits new request\n");
        /*
        const char *keepopen="[KEEPING CONNECTION OPEN]";
        storerequest((char *)keepopen, strlen(keepopen));
        */
      }
      /* if we got a CONNECT, loop and get another request as well! */
    } while(req.open || (req.testno == DOCNUMBER_CONNECT));

    if(got_exit_signal)
      break;

    logmsg("====> Client disconnect");

    if(!req.open)
      /* When instructed to close connection after server-reply we
         wait a very small amount of time before doing so. If this
         is not done client might get an ECONNRESET before reading
         a single byte of server-reply. */
      wait_ms(50);

    if(msgsock != CURL_SOCKET_BAD) {
      sclose(msgsock);
      msgsock = CURL_SOCKET_BAD;
    }

    if(serverlogslocked) {
      serverlogslocked = 0;
      clear_advisor_read_lock(SERVERLOGS_LOCK);
    }

    if (req.testno == DOCNUMBER_QUIT)
      break;
#ifdef CURL_SWS_FORK_ENABLED
    }
#endif
  }

sws_cleanup:

  if((msgsock != sock) && (msgsock != CURL_SOCKET_BAD))
    sclose(msgsock);

  if(sock != CURL_SOCKET_BAD)
    sclose(sock);

  if(got_exit_signal)
    logmsg("signalled to die");

  if(wrotepidfile)
    unlink(pidname);

  if(serverlogslocked) {
    serverlogslocked = 0;
    clear_advisor_read_lock(SERVERLOGS_LOCK);
  }

  restore_signal_handlers();

  if(got_exit_signal) {
    logmsg("========> %s sws (port: %d pid: %ld) exits with signal (%d)",
           ipv_inuse, (int)port, pid, exit_signal);
    /*
     * To properly set the return status of the process we
     * must raise the same signal SIGINT or SIGTERM that we
     * caught and let the old handler take care of it.
     */
    raise(exit_signal);
  }

  logmsg("========> sws quits");
  return 0;
}

