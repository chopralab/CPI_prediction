#!/usr/bin/env perl -w

use warnings;
use strict;

my $all_results = shift;
my $compounds = shift;
my $proteins = shift;

open my $results_fh, '<', $all_results or die "$!\n";
open my $compounds_fh, '<', $compounds or die "$!\n";

while (my $compound = <$compounds_fh>) {
    chomp $compound;
    open my $proteins_fh, '<', $proteins or die "$!\n";
    while (my $protein = <$proteins_fh>) {
        chomp $protein;
        my $result = <$results_fh>;
        print "$compound\t$protein\t$result";
    }
    close $proteins_fh;
}

close $compounds_fh;
close $results_fh;

