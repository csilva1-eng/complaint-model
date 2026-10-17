package com.cie.web;

import com.cie.model.Models.Ticket;
import com.cie.model.Models.TicketEdit;
import com.cie.tickets.TicketService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * The human-review workflow, one endpoint per step:
 *
 *   POST /api/tickets?clusterId=   draft from a cluster (returns the existing ticket if one is open)
 *   GET  /api/tickets              list;   GET /api/tickets/{id}  one ticket
 *   PUT  /api/tickets/{id}         analyst edits the draft
 *   POST /api/tickets/{id}/approve explicit approval -> creates the Jira issue
 *   POST /api/tickets/{id}/reject  discard the draft
 */
@RestController
@RequestMapping("/api/tickets")
public class TicketController {
    private final TicketService tickets;

    public TicketController(TicketService tickets) {
        this.tickets = tickets;
    }

    @PostMapping public Ticket draft(@RequestParam int clusterId) { return tickets.createDraft(clusterId); }
    @GetMapping public List<Ticket> list() { return tickets.list(); }
    @GetMapping("/{id}") public Ticket get(@PathVariable String id) { return tickets.get(id); }
    @PutMapping("/{id}") public Ticket edit(@PathVariable String id, @RequestBody TicketEdit edit) { return tickets.update(id, edit); }
    @PostMapping("/{id}/approve") public Ticket approve(@PathVariable String id) { return tickets.approve(id); }
    @PostMapping("/{id}/reject") public Ticket reject(@PathVariable String id) { return tickets.reject(id); }
}
