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
 * RFC1939 POP3 protocol
 * RFC2384 POP URL Scheme
 * RFC2595 Using TLS with IMAP, POP3 and ACAP
 *
 ***************************************************************************/

#include "setup.h"

#ifndef CURL_DISABLE_POP3

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
#ifdef HAVE_UTSNAME_H
#include <sys/utsname.h>
#endif
#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif
#ifdef __VMS
#include <in.h>
#include <inet.h>
#endif

#if (defined(NETWARE) && defined(__NOVELL_LIBC__))
#undef in_addr_t
#define in_addr_t unsigned long
#endif

#include <curl/curl.h>
#include "urldata.h"
#include "sendf.h"
#include "if2ip.h"
#include "hostip.h"
#include "progress.h"
#include "transfer.h"
#include "escape.h"
#include "http.h" /* for HTTP proxy tunnel stuff */
#include "socks.h"
#include "pop3.h"

#include "strtoofft.h"
#include "strequal.h"
#include "sslgen.h"
#include "connect.h"
#include "strerror.h"
#include "select.h"
#include "multiif.h"
#include "url.h"
#include "rawstr.h"
#include "strtoofft.h"

#define _MPRINTF_REPLACE /* use our functions only */
#include <curl/mprintf.h>

#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

/* Local API functions */
static CURLcode pop3_parse_url_path(struct connectdata *conn);
static CURLcode pop3_parse_custom_request(struct connectdata *conn);
static CURLcode pop3_regular_transfer(struct connectdata *conn, bool *done);
static CURLcode pop3_do(struct connectdata *conn, bool *done);
static CURLcode pop3_done(struct connectdata *conn,
                          CURLcode, bool premature);
static CURLcode pop3_connect(struct connectdata *conn, bool *done);
static CURLcode pop3_disconnect(struct connectdata *conn, bool dead);
static CURLcode pop3_multi_statemach(struct connectdata *conn, bool *done);
static int pop3_getsock(struct connectdata *conn,
                        curl_socket_t *socks,
                        int numsocks);
static CURLcode pop3_doing(struct connectdata *conn,
                           bool *dophase_done);
static CURLcode pop3_setup_connection(struct connectdata * conn);

/*
 * POP3 protocol handler.
 */

const struct Curl_handler Curl_handler_pop3 = {
  "POP3",                           /* scheme */
  pop3_setup_connection,            /* setup_connection */
  pop3_do,                          /* do_it */
  pop3_done,                        /* done */
  ZERO_NULL,                        /* do_more */
  pop3_connect,                     /* connect_it */
  pop3_multi_statemach,             /* connecting */
  pop3_doing,                       /* doing */
  pop3_getsock,                     /* proto_getsock */
  pop3_getsock,                     /* doing_getsock */
  ZERO_NULL,                        /* domore_getsock */
  ZERO_NULL,                        /* perform_getsock */
  pop3_disconnect,                  /* disconnect */
  ZERO_NULL,                        /* readwrite */
  PORT_POP3,                        /* defport */
  CURLPROTO_POP3,                   /* protocol */
  PROTOPT_CLOSEACTION | PROTOPT_NOURLQUERY /* flags */
};

#ifdef USE_SSL
/*
 * POP3S protocol handler.
 */

const struct Curl_handler Curl_handler_pop3s = {
  "POP3S",                          /* scheme */
  pop3_setup_connection,            /* setup_connection */
  pop3_do,                          /* do_it */
  pop3_done,                        /* done */
  ZERO_NULL,                        /* do_more */
  pop3_connect,                     /* connect_it */
  pop3_multi_statemach,             /* connecting */
  pop3_doing,                       /* doing */
  pop3_getsock,                     /* proto_getsock */
  pop3_getsock,                     /* doing_getsock */
  ZERO_NULL,                        /* domore_getsock */
  ZERO_NULL,                        /* perform_getsock */
  pop3_disconnect,                  /* disconnect */
  ZERO_NULL,                        /* readwrite */
  PORT_POP3S,                       /* defport */
  CURLPROTO_POP3 | CURLPROTO_POP3S, /* protocol */
  PROTOPT_CLOSEACTION | PROTOPT_SSL
  | PROTOPT_NOURLQUERY              /* flags */
};
#endif

#ifndef CURL_DISABLE_HTTP
/*
 * HTTP-proxyed POP3 protocol handler.
 */

static const struct Curl_handler Curl_handler_pop3_proxy = {
  "POP3",                               /* scheme */
  ZERO_NULL,                            /* setup_connection */
  Curl_http,                            /* do_it */
  Curl_http_done,                       /* done */
  ZERO_NULL,                            /* do_more */
  ZERO_NULL,                            /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  ZERO_NULL,                            /* disconnect */
  ZERO_NULL,                            /* readwrite */
  PORT_POP3,                            /* defport */
  CURLPROTO_HTTP,                       /* protocol */
  PROTOPT_NONE                          /* flags */
};

#ifdef USE_SSL
/*
 * HTTP-proxyed POP3S protocol handler.
 */

static const struct Curl_handler Curl_handler_pop3s_proxy = {
  "POP3S",                              /* scheme */
  ZERO_NULL,                            /* setup_connection */
  Curl_http,                            /* do_it */
  Curl_http_done,                       /* done */
  ZERO_NULL,                            /* do_more */
  ZERO_NULL,                            /* connect_it */
  ZERO_NULL,                            /* connecting */
  ZERO_NULL,                            /* doing */
  ZERO_NULL,                            /* proto_getsock */
  ZERO_NULL,                            /* doing_getsock */
  ZERO_NULL,                            /* domore_getsock */
  ZERO_NULL,                            /* perform_getsock */
  ZERO_NULL,                            /* disconnect */
  ZERO_NULL,                            /* readwrite */
  PORT_POP3S,                           /* defport */
  CURLPROTO_HTTP,                       /* protocol */
  PROTOPT_NONE                          /* flags */
};
#endif
#endif

/* function that checks for a pop3 status code at the start of the given
   string */
static int pop3_endofresp(struct pingpong *pp,
                          int *resp)
{
  char *line = pp->linestart_resp;
  size_t len = pp->nread_resp;

  if(((len >= 3) && !memcmp("+OK", line, 3)) ||
     ((len >= 4) && !memcmp("-ERR", line, 4))) {
    *resp = line[1]; /* O or E */
    return TRUE;
  }

  return FALSE; /* nothing for us */
}

/* This is the ONLY way to change POP3 state! */
static void state(struct connectdata *conn,
                  pop3state newstate)
{
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  /* for debug purposes */
  static const char * const names[]={
    "STOP",
    "SERVERGREET",
    "USER",
    "PASS",
    "STARTTLS",
    "COMMAND",
    "QUIT",
    /* LAST */
  };
#endif
  struct pop3_conn *pop3c = &conn->proto.pop3c;
#if defined(DEBUGBUILD) && !defined(CURL_DISABLE_VERBOSE_STRINGS)
  if(pop3c->state != newstate)
    infof(conn->data, "POP3 %p state change from %s to %s\n",
          pop3c, names[pop3c->state], names[newstate]);
#endif
  pop3c->state = newstate;
}

static CURLcode pop3_state_user(struct connectdata *conn)
{
  CURLcode result;
  struct FTP *pop3 = conn->data->state.proto.pop3;

  /* send USER */
  result = Curl_pp_sendf(&conn->proto.pop3c.pp, "USER %s",
                         pop3->user ? pop3->user : "");
  if(result)
    return result;

  state(conn, POP3_USER);

  return CURLE_OK;
}

/* For the POP3 "protocol connect" and "doing" phases only */
static int pop3_getsock(struct connectdata *conn,
                        curl_socket_t *socks,
                        int numsocks)
{
  return Curl_pp_getsock(&conn->proto.pop3c.pp, socks, numsocks);
}

#ifdef USE_SSL
static void pop3_to_pop3s(struct connectdata *conn)
{
  conn->handler = &Curl_handler_pop3s;
}
#else
#define pop3_to_pop3s(x) Curl_nop_stmt
#endif

/* for the initial server greeting */
static CURLcode pop3_state_servergreet_resp(struct connectdata *conn,
                                            int pop3code,
                                            pop3state instate)
{
  CURLcode result = CURLE_OK;
  struct SessionHandle *data = conn->data;
  struct pop3_conn *pop3c = &conn->proto.pop3c;

  (void)instate; /* no use for this yet */

  if(pop3code != 'O') {
    failf(data, "Got unexpected pop3-server response");
    return CURLE_FTP_WEIRD_SERVER_REPLY;
  }

  if(data->set.use_ssl && !conn->ssl[FIRSTSOCKET].use) {
    /* We don't have a SSL/TLS connection yet, but SSL is requested. Switch
       to TLS connection now */
    result = Curl_pp_sendf(&pop3c->pp, "STLS");
    state(conn, POP3_STARTTLS);
  }
  else
    result = pop3_state_user(conn);

  return result;
}

/* for STARTTLS responses */
static CURLcode pop3_state_starttls_resp(struct connectdata *conn,
                                         int pop3code,
                                         pop3state instate)
{
  CURLcode result = CURLE_OK;
  struct SessionHandle *data = conn->data;

  (void)instate; /* no use for this yet */

  if(pop3code != 'O') {
    if(data->set.use_ssl != CURLUSESSL_TRY) {
      failf(data, "STARTTLS denied. %c", pop3code);
      result = CURLE_USE_SSL_FAILED;
      state(conn, POP3_STOP);
    }
    else
      result = pop3_state_user(conn);
  }
  else {
    /* Curl_ssl_connect is BLOCKING */
    result = Curl_ssl_connect(conn, FIRSTSOCKET);
    if(CURLE_OK == result) {
      pop3_to_pop3s(conn);
      result = pop3_state_user(conn);
    }
    else {
      state(conn, POP3_STOP);
    }
  }

  return result;
}

/* for USER responses */
static CURLcode pop3_state_user_resp(struct connectdata *conn,
                                     int pop3code,
                                     pop3state instate)
{
  CURLcode result = CURLE_OK;
  struct SessionHandle *data = conn->data;
  struct FTP *pop3 = data->state.proto.pop3;

  (void)instate; /* no use for this yet */

  if(pop3code != 'O') {
    failf(data, "Access denied. %c", pop3code);
    result = CURLE_LOGIN_DENIED;
  }
  else
    /* send PASS */
    result = Curl_pp_sendf(&conn->proto.pop3c.pp, "PASS %s",
                           pop3->passwd ? pop3->passwd : "");
  if(result)
    return result;

  state(conn, POP3_PASS);

  return result;
}

/* for PASS responses */
static CURLcode pop3_state_pass_resp(struct connectdata *conn,
                                     int pop3code,
                                     pop3state instate)
{
  CURLcode result = CURLE_OK;
  struct SessionHandle *data = conn->data;

  (void)instate; /* no use for this yet */

  if(pop3code != 'O') {
    failf(data, "Access denied. %c", pop3code);
    result = CURLE_LOGIN_DENIED;
  }

  state(conn, POP3_STOP);

  return result;
}

/* for the command response */
static CURLcode pop3_state_command_resp(struct connectdata *conn,
                                        int pop3code,
                                        pop3state instate)
{
  CURLcode result = CURLE_OK;
  struct SessionHandle *data = conn->data;
  struct FTP *pop3 = data->state.proto.pop3;
  struct pop3_conn *pop3c = &conn->proto.pop3c;
  struct pingpong *pp = &pop3c->pp;

  (void)instate; /* no use for this yet */

  if('O' != pop3code) {
    state(conn, POP3_STOP);
    return CURLE_RECV_ERROR;
  }

  /* This 'OK' line ends with a CR LF pair which is the two first bytes of the
     EOB string so count this is two matching bytes. This is necessary to make
     the code detect the EOB if the only data than comes now is %2e CR LF like
     when there is no body to return. */
  pop3c->eob = 2;

  /* But since this initial CR LF pair is not part of the actual body, we set
     the strip counter here so that these bytes won't be delivered. */
  pop3c->strip = 2;

  /* POP3 download */
  Curl_setup_transfer(conn, FIRSTSOCKET, -1, FALSE, pop3->bytecountp,
                      -1, NULL); /* no upload here */

  if(pp->cache) {
    /* The header "cache" contains a bunch of data that is actually body
       content so send it as such. Note that there may even be additional
       "headers" after the body */

    if(!data->set.opt_no_body) {
      result = Curl_pop3_write(conn, pp->cache, pp->cache_size);
      if(result)
        return result;
    }

    /* Free the cache */
    Curl_safefree(pp->cache);

    /* Reset the cache size */
    pp->cache_size = 0;
  }

  state(conn, POP3_STOP);

  return result;
}

/* start the DO phase for the command */
static CURLcode pop3_command(struct connectdata *conn)
{
  CURLcode result = CURLE_OK;
  struct pop3_conn *pop3c = &conn->proto.pop3c;
  const char *command = NULL;

  /* Calculate the default command */
  if(pop3c->mailbox[0] == '\0' || conn->data->set.ftp_list_only) {
    command = "LIST";

    if(pop3c->mailbox[0] != '\0') {
      /* Message specific LIST so skip the BODY transfer */
      struct FTP *pop3 = conn->data->state.proto.pop3;
      pop3->transfer = FTPTRANSFER_INFO;
    }
  }
  else
    command = "RETR";

  /* Send the command */
  if(pop3c->mailbox[0] != '\0')
    result = Curl_pp_sendf(&conn->proto.pop3c.pp, "%s %s",
                           (pop3c->custom && pop3c->custom[0] != '\0' ?
                            pop3c->custom : command), pop3c->mailbox);
  else
    result = Curl_pp_sendf(&conn->proto.pop3c.pp,
                           (pop3c->custom && pop3c->custom[0] != '\0' ?
                            pop3c->custom : command));

  if(result)
    return result;

  state(conn, POP3_COMMAND);

  return result;
}

static CURLcode pop3_statemach_act(struct connectdata *conn)
{
  CURLcode result;
  curl_socket_t sock = conn->sock[FIRSTSOCKET];
  int pop3code;
  struct pop3_conn *pop3c = &conn->proto.pop3c;
  struct pingpong *pp = &pop3c->pp;
  size_t nread = 0;

  if(pp->sendleft)
    return Curl_pp_flushsend(pp);

  /* we read a piece of response */
  result = Curl_pp_readresp(sock, pp, &pop3code, &nread);
  if(result)
    return result;

  if(pop3code) {
    /* we have now received a full POP3 server response */
    switch(pop3c->state) {
    case POP3_SERVERGREET:
      result = pop3_state_servergreet_resp(conn, pop3code, pop3c->state);
      break;

    case POP3_USER:
      result = pop3_state_user_resp(conn, pop3code, pop3c->state);
      break;

    case POP3_PASS:
      result = pop3_state_pass_resp(conn, pop3code, pop3c->state);
      break;

    case POP3_STARTTLS:
      result = pop3_state_starttls_resp(conn, pop3code, pop3c->state);
      break;

    case POP3_COMMAND:
      result = pop3_state_command_resp(conn, pop3code, pop3c->state);
      break;

    case POP3_QUIT:
      /* fallthrough, just stop! */
    default:
      /* internal error */
      state(conn, POP3_STOP);
      break;
    }
  }

  return result;
}

/* called repeatedly until done from multi.c */
static CURLcode pop3_multi_statemach(struct connectdata *conn, bool *done)
{
  struct pop3_conn *pop3c = &conn->proto.pop3c;
  CURLcode result = Curl_pp_multi_statemach(&pop3c->pp);

  *done = (pop3c->state == POP3_STOP) ? TRUE : FALSE;

  return result;
}

static CURLcode pop3_easy_statemach(struct connectdata *conn)
{
  struct pop3_conn *pop3c = &conn->proto.pop3c;
  struct pingpong *pp = &pop3c->pp;
  CURLcode result = CURLE_OK;

  while(pop3c->state != POP3_STOP) {
    result = Curl_pp_easy_statemach(pp);
    if(result)
      break;
  }

  return result;
}

/*
 * Allocate and initialize the struct POP3 for the current SessionHandle.  If
 * need be.
 */
static CURLcode pop3_init(struct connectdata *conn)
{
  struct SessionHandle *data = conn->data;
  struct FTP *pop3 = data->state.proto.pop3;

  if(!pop3) {
    pop3 = data->state.proto.pop3 = calloc(sizeof(struct FTP), 1);
    if(!pop3)
      return CURLE_OUT_OF_MEMORY;
  }

  /* get some initial data into the pop3 struct */
  pop3->bytecountp = &data->req.bytecount;

  /* No need to duplicate user+password, the connectdata struct won't change
     during a session, but we re-init them here since on subsequent inits
     since the conn struct may have changed or been replaced.
  */
  pop3->user = conn->user;
  pop3->passwd = conn->passwd;

  return CURLE_OK;
}

/*
 * pop3_connect() should do everything that is to be considered a part of
 * the connection phase.
 *
 * The variable 'done' points to will be TRUE if the protocol-layer connect
 * phase is done when this function returns, or FALSE is not. When called as
 * a part of the easy interface, it will always be TRUE.
 */
static CURLcode pop3_connect(struct connectdata *conn,
                             bool *done) /* see description above */
{
  CURLcode result;
  struct pop3_conn *pop3c = &conn->proto.pop3c;
  struct SessionHandle *data = conn->data;
  struct pingpong *pp = &pop3c->pp;

  *done = FALSE; /* default to not done yet */

  /* If there already is a protocol-specific struct allocated for this
     sessionhandle, deal with it */
  Curl_reset_reqproto(conn);

  result = pop3_init(conn);
  if(CURLE_OK != result)
    return result;

  /* We always support persistent connections on pop3 */
  conn->bits.close = FALSE;

  pp->response_time = RESP_TIMEOUT; /* set default response time-out */
  pp->statemach_act = pop3_statemach_act;
  pp->endofresp = pop3_endofresp;
  pp->conn = conn;

  if(conn->handler->flags & PROTOPT_SSL) {
    /* BLOCKING */
    result = Curl_ssl_connect(conn, FIRSTSOCKET);
    if(result)
      return result;
  }

  Curl_pp_init(pp); /* init the response reader stuff */

  /* When we connect, we start in the state where we await the server greet
     response */
  state(conn, POP3_SERVERGREET);

  if(data->state.used_interface == Curl_if_multi)
    result = pop3_multi_statemach(conn, done);
  else {
    result = pop3_easy_statemach(conn);
    if(!result)
      *done = TRUE;
  }

  return result;
}

/***********************************************************************
 *
 * pop3_done()
 *
 * The DONE function. This does what needs to be done after a single DO has
 * performed.
 *
 * Input argument is already checked for validity.
 */
static CURLcode pop3_done(struct connectdata *conn, CURLcode status,
                          bool premature)
{
  struct SessionHandle *data = conn->data;
  struct FTP *pop3 = data->state.proto.pop3;
  struct pop3_conn *pop3c = &conn->proto.pop3c;
  CURLcode result = CURLE_OK;

  (void)premature;

  if(!pop3)
    /* When the easy handle is removed from the multi while libcurl is still
     * trying to resolve the host name, it seems that the pop3 struct is not
     * yet initialized, but the removal action calls Curl_done() which calls
     * this function. So we simply return success if no pop3 pointer is set.
     */
    return CURLE_OK;

  if(status) {
    conn->bits.close = TRUE; /* marked for closure */
    result = status;         /* use the already set error code */
  }

  /* Clear our variables for the next connection */
  Curl_safefree(pop3c->mailbox);
  Curl_safefree(pop3c->custom);

  /* Clear the transfer mode for the next connection */
  pop3->transfer = FTPTRANSFER_BODY;

  return result;
}

/***********************************************************************
 *
 * pop3_perform()
 *
 * This is the actual DO function for POP3. Get a file/directory according to
 * the options previously setup.
 */

static
CURLcode pop3_perform(struct connectdata *conn,
                     bool *connected,  /* connect status after PASV / PORT */
                     bool *dophase_done)
{
  /* this is POP3 and no proxy */
  CURLcode result = CURLE_OK;

  DEBUGF(infof(conn->data, "DO phase starts\n"));

  if(conn->data->set.opt_no_body) {
    /* requested no body means no transfer... */
    struct FTP *pop3 = conn->data->state.proto.pop3;
    pop3->transfer = FTPTRANSFER_INFO;
  }

  *dophase_done = FALSE; /* not done yet */

  /* start the first command in the DO phase */
  result = pop3_command(conn);
  if(result)
    return result;

  /* run the state-machine */
  if(conn->data->state.used_interface == Curl_if_multi)
    result = pop3_multi_statemach(conn, dophase_done);
  else {
    result = pop3_easy_statemach(conn);
    *dophase_done = TRUE; /* with the easy interface we are done here */
  }
  *connected = conn->bits.tcpconnect[FIRSTSOCKET];

  if(*dophase_done)
    DEBUGF(infof(conn->data, "DO phase is complete\n"));

  return result;
}

/***********************************************************************
 *
 * pop3_do()
 *
 * This function is registered as 'curl_do' function. It decodes the path
 * parts etc as a wrapper to the actual DO function (pop3_perform).
 *
 * The input argument is already checked for validity.
 */
static CURLcode pop3_do(struct connectdata *conn, bool *done)
{
  CURLcode retcode = CURLE_OK;

  *done = FALSE; /* default to false */

  /*
    Since connections can be re-used between SessionHandles, this might be a
    connection already existing but on a fresh SessionHandle struct so we must
    make sure we have a good 'struct POP3' to play with. For new connections,
    the struct POP3 is allocated and setup in the pop3_connect() function.
  */
  Curl_reset_reqproto(conn);
  retcode = pop3_init(conn);
  if(retcode)
    return retcode;

  /* Parse the URL path */
  retcode = pop3_parse_url_path(conn);
  if(retcode)
    return retcode;

  /* Parse the custom request */
  retcode = pop3_parse_custom_request(conn);
  if(retcode)
    return retcode;

  retcode = pop3_regular_transfer(conn, done);

  return retcode;
}

/***********************************************************************
 *
 * pop3_quit()
 *
 * This should be called before calling sclose().  We should then wait for the
 * response from the server before returning. The calling code should then try
 * to close the connection.
 *
 */
static CURLcode pop3_quit(struct connectdata *conn)
{
  CURLcode result = CURLE_OK;

  result = Curl_pp_sendf(&conn->proto.pop3c.pp, "QUIT", NULL);
  if(result)
    return result;

  state(conn, POP3_QUIT);

  result = pop3_easy_statemach(conn);

  return result;
}

/***********************************************************************
 *
 * pop3_disconnect()
 *
 * Disconnect from an POP3 server. Cleanup protocol-specific per-connection
 * resources. BLOCKING.
 */
static CURLcode pop3_disconnect(struct connectdata *conn, bool dead_connection)
{
  struct pop3_conn *pop3c= &conn->proto.pop3c;

  /* We cannot send quit unconditionally. If this connection is stale or
     bad in any way, sending quit and waiting around here will make the
     disconnect wait in vain and cause more problems than we need to.
  */

  /* The POP3 session may or may not have been allocated/setup at this
     point! */
  if(!dead_connection && pop3c->pp.conn)
    (void)pop3_quit(conn); /* ignore errors on the LOGOUT */

  Curl_pp_disconnect(&pop3c->pp);

  return CURLE_OK;
}

/***********************************************************************
 *
 * pop3_parse_url_path()
 *
 * Parse the URL path into separate path components.
 *
 */
static CURLcode pop3_parse_url_path(struct connectdata *conn)
{
  /* the pop3 struct is already inited in pop3_connect() */
  struct pop3_conn *pop3c = &conn->proto.pop3c;
  struct SessionHandle *data = conn->data;
  const char *path = data->state.path;

  /* URL decode the path and use this mailbox */
  return Curl_urldecode(data, path, 0, &pop3c->mailbox, NULL, TRUE);
}

static CURLcode pop3_parse_custom_request(struct connectdata *conn)
{
  CURLcode result = CURLE_OK;
  struct pop3_conn *pop3c = &conn->proto.pop3c;
  struct SessionHandle *data = conn->data;
  const char *custom = conn->data->set.str[STRING_CUSTOMREQUEST];

  /* URL decode the custom request */
  if(custom)
    result = Curl_urldecode(data, custom, 0, &pop3c->custom, NULL, TRUE);

  return result;
}

/* call this when the DO phase has completed */
static CURLcode pop3_dophase_done(struct connectdata *conn,
                                  bool connected)
{
  struct FTP *pop3 = conn->data->state.proto.pop3;

  (void)connected;

  if(pop3->transfer != FTPTRANSFER_BODY)
    /* no data to transfer */
    Curl_setup_transfer(conn, -1, -1, FALSE, NULL, -1, NULL);

  return CURLE_OK;
}

/* called from multi.c while DOing */
static CURLcode pop3_doing(struct connectdata *conn,
                               bool *dophase_done)
{
  CURLcode result;
  result = pop3_multi_statemach(conn, dophase_done);

  if(*dophase_done) {
    result = pop3_dophase_done(conn, FALSE /* not connected */);

    DEBUGF(infof(conn->data, "DO phase is complete\n"));
  }

  return result;
}

/***********************************************************************
 *
 * pop3_regular_transfer()
 *
 * The input argument is already checked for validity.
 *
 * Performs all commands done before a regular transfer between a local and a
 * remote host.
 *
 */
static CURLcode pop3_regular_transfer(struct connectdata *conn,
                                      bool *dophase_done)
{
  CURLcode result = CURLE_OK;
  bool connected = FALSE;
  struct SessionHandle *data = conn->data;
  data->req.size = -1; /* make sure this is unknown at this point */

  Curl_pgrsSetUploadCounter(data, 0);
  Curl_pgrsSetDownloadCounter(data, 0);
  Curl_pgrsSetUploadSize(data, 0);
  Curl_pgrsSetDownloadSize(data, 0);

  result = pop3_perform(conn,
                        &connected,    /* have we connected after PASV/PORT */
                        dophase_done); /* all commands in the DO-phase done? */

  if(CURLE_OK == result) {

    if(!*dophase_done)
      /* the DO phase has not completed yet */
      return CURLE_OK;

    result = pop3_dophase_done(conn, connected);
    if(result)
      return result;
  }

  return result;
}

static CURLcode pop3_setup_connection(struct connectdata * conn)
{
  struct SessionHandle *data = conn->data;

  if(conn->bits.httpproxy && !data->set.tunnel_thru_httpproxy) {
    /* Unless we have asked to tunnel pop3 operations through the proxy, we
       switch and use HTTP operations only */
#ifndef CURL_DISABLE_HTTP
    if(conn->handler == &Curl_handler_pop3)
      conn->handler = &Curl_handler_pop3_proxy;
    else {
#ifdef USE_SSL
      conn->handler = &Curl_handler_pop3s_proxy;
#else
      failf(data, "POP3S not supported!");
      return CURLE_UNSUPPORTED_PROTOCOL;
#endif
    }

    /*
     * We explicitly mark this connection as persistent here as we're doing
     * POP3 over HTTP and thus we accidentally avoid setting this value
     * otherwise.
     */
    conn->bits.close = FALSE;
#else
    failf(data, "POP3 over http proxy requires HTTP support built-in!");
    return CURLE_UNSUPPORTED_PROTOCOL;
#endif
  }

  data->state.path++;   /* don't include the initial slash */

  return CURLE_OK;
}

/* this is the 5-bytes End-Of-Body marker for POP3 */
#define POP3_EOB "\x0d\x0a\x2e\x0d\x0a"
#define POP3_EOB_LEN 5

/*
 * This function scans the body after the end-of-body and writes everything
 * until the end is found.
 */
CURLcode Curl_pop3_write(struct connectdata *conn,
                         char *str,
                         size_t nread)
{
  /* This code could be made into a special function in the handler struct */
  CURLcode result = CURLE_OK;
  struct SessionHandle *data = conn->data;
  struct SingleRequest *k = &data->req;

  struct pop3_conn *pop3c = &conn->proto.pop3c;
  bool strip_dot = FALSE;
  size_t last = 0;
  size_t i;

  /* Search through the buffer looking for the end-of-body marker which is
     5 bytes (0d 0a 2e 0d 0a). Note that a line starting with a dot matches
     the eob so the server will have prefixed it with an extra dot which we
     need to strip out. Additionally the marker could of course be spread out
     over 5 different data chunks */
  for(i = 0; i < nread; i++) {
    size_t prev = pop3c->eob;

    switch(str[i]) {
    case 0x0d:
      if(pop3c->eob == 0) {
        pop3c->eob++;

        if(i) {
          /* Write out the body part that didn't match */
          result = Curl_client_write(conn, CLIENTWRITE_BODY, &str[last],
                                     i - last);

          if(result)
            return result;

          last = i;
        }
      }
      else if(pop3c->eob == 3)
        pop3c->eob++;
      else
        /* If the character match wasn't at position 0 or 3 then restart the
           pattern matching */
        pop3c->eob = 1;
      break;

    case 0x0a:
      if(pop3c->eob == 1 || pop3c->eob == 4)
        pop3c->eob++;
      else
        /* If the character match wasn't at position 1 or 4 then start the
           search again */
        pop3c->eob = 0;
      break;

    case 0x2e:
      if(pop3c->eob == 2)
        pop3c->eob++;
      else if(pop3c->eob == 3) {
        /* We have an extra dot after the CRLF which we need to strip off */
        strip_dot = TRUE;
        pop3c->eob = 0;
      }
      else
        /* If the character match wasn't at position 2 then start the search
           again */
        pop3c->eob = 0;
      break;

    default:
      pop3c->eob = 0;
      break;
    }

    /* Did we have a partial match which has subsequently failed? */
    if(prev && prev >= pop3c->eob) {
      /* Strip can only be non-zero for the very first mismatch after CRLF
         and then both prev and strip are equal and nothing will be output
         below */
      while(prev && pop3c->strip) {
        prev--;
        pop3c->strip--;
      }

      if(prev) {
        /* If the partial match was the CRLF and dot then only write the CRLF
           as the server would have inserted the dot */
        result = Curl_client_write(conn, CLIENTWRITE_BODY, (char*)POP3_EOB,
                                   strip_dot ? prev - 1 : prev);

        if(result)
          return result;

        last = i;
        strip_dot = FALSE;
      }
    }
  }

  if(pop3c->eob == POP3_EOB_LEN) {
    /* We have a full match so the transfer is done, however we must transfer
    the CRLF at the start of the EOB as this is considered to be part of the
    message as per RFC-1939, sect. 3 */
    result = Curl_client_write(conn, CLIENTWRITE_BODY, (char*)POP3_EOB, 2);

    k->keepon &= ~KEEP_RECV;
    pop3c->eob = 0;

    return result;
  }

  if(pop3c->eob)
    /* While EOB is matching nothing should be output */
    return CURLE_OK;

  if(nread - last) {
    result = Curl_client_write(conn, CLIENTWRITE_BODY, &str[last],
                               nread - last);
  }

  return result;
}

#endif /* CURL_DISABLE_POP3 */
